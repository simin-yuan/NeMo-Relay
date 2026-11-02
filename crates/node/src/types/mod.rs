// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod resource_metrics;

use std::sync::Arc;

use napi::Env;
use napi::bindgen_prelude::Unknown;
use napi_derive::napi;

use crate::error::{Error, Result};
use nemo_relay_runtime::api::runtime::subscriber_dispatcher::PublicationBuffer;
use nemo_relay_runtime::api::runtime::{ScopeStackHandle, create_scope_stack};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use nemo_relay_codecs::resolve::ProviderSurface;
use nemo_relay_observability::otel_metrics::MetricTemporality as CoreMetricTemporality;
use nemo_relay_runtime::api::event::{
    DataSchema as CoreDataSchema, LogSeverity as CoreLogSeverity, MetricKind as CoreMetricKind,
    MetricMeasurement as CoreMetricMeasurement, MetricValueType as CoreMetricValueType,
};
use nemo_relay_runtime::api::llm::{LlmHandle as CoreLlmHandle, LlmRequest as CoreLlmRequest};
use nemo_relay_runtime::api::scope::{ScopeHandle as CoreScopeHandle, ScopeType as CoreScopeType};
use nemo_relay_runtime::api::tool::{
    ToolExecutionResult as CoreToolExecutionResult, ToolHandle as CoreToolHandle,
};
use nemo_relay_types::codec::request::AnnotatedLlmRequest;
use nemo_relay_types::codec::traits::{LlmCodec, LlmResponseCodec};

macro_rules! mirror_variants {
    ($local:ident <=> $core:ident { $($variant:ident),+ $(,)? }) => {
        mirror_variants!($local => $core { $($variant),+ });

        impl From<$core> for $local {
            fn from(value: $core) -> Self {
                match value {
                    $($core::$variant => Self::$variant,)+
                }
            }
        }
    };
    ($local:ident => $core:ident { $($variant:ident),+ $(,)? }) => {
        impl From<$local> for $core {
            fn from(value: $local) -> Self {
                match value {
                    $($local::$variant => Self::$variant,)+
                }
            }
        }
    };
}

pub(crate) use mirror_variants;

/// Identity of one visible Relay scope.
#[napi(object)]
pub struct ScopeStackEntry {
    /// Canonical UUID of the scope handle.
    pub scope_id: String,
    /// Existing Relay scope category.
    pub scope_type: ScopeType,
}

/// Detached root-to-top values including implicit and synthetic roots.
#[napi(object)]
pub struct ScopeStackSnapshot {
    /// Ordered owned entries from one coherent read.
    pub scopes: Vec<ScopeStackEntry>,
}

impl From<nemo_relay_runtime::api::runtime::ScopeStackSnapshot> for ScopeStackSnapshot {
    fn from(snapshot: nemo_relay_runtime::api::runtime::ScopeStackSnapshot) -> Self {
        Self {
            scopes: snapshot
                .scopes
                .into_iter()
                .map(|scope| ScopeStackEntry {
                    scope_id: scope.scope_id.to_string(),
                    scope_type: scope.scope_type.into(),
                })
                .collect(),
        }
    }
}

/// The type of an execution scope in the agent runtime hierarchy.
#[napi]
pub enum ScopeType {
    /// An autonomous agent scope.
    Agent,
    /// A generic function invocation scope.
    Function,
    /// A tool execution scope.
    Tool,
    /// A large language model call scope.
    Llm,
    /// A retriever (vector search / RAG) scope.
    Retriever,
    /// An embedding model scope.
    Embedder,
    /// A reranker model scope.
    Reranker,
    /// A guardrail evaluation scope.
    Guardrail,
    /// An evaluator / scoring scope.
    Evaluator,
    /// A user-defined custom scope type.
    Custom,
    /// An unknown or unclassified scope type.
    Unknown,
}

mirror_variants!(ScopeType <=> CoreScopeType {
    Agent, Function, Tool, Llm, Retriever, Embedder, Reranker, Guardrail, Evaluator, Custom,
    Unknown,
});

/// Severity attached to a mark projected as an OpenTelemetry log.
#[napi]
pub enum LogSeverity {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

mirror_variants!(LogSeverity => CoreLogSeverity { Trace, Debug, Info, Warn, Error });

/// OpenTelemetry instrument kind recorded by a metric measurement.
#[napi]
pub enum MetricKind {
    Counter,
    UpDownCounter,
    Gauge,
    Histogram,
}

mirror_variants!(MetricKind => CoreMetricKind { Counter, UpDownCounter, Gauge, Histogram });

/// Explicit numeric representation used by a metric measurement.
#[napi]
pub enum MetricValueType {
    U64,
    I64,
    F64,
}

mirror_variants!(MetricValueType => CoreMetricValueType { U64, I64, F64 });

/// Preferred aggregation temporality for OTLP metric export.
#[napi]
pub enum MetricTemporality {
    Cumulative,
    Delta,
    LowMemory,
}

mirror_variants!(MetricTemporality => CoreMetricTemporality { Cumulative, Delta, LowMemory });

/// Schema identifier attached to an event's opaque data payload.
#[napi(object)]
pub struct DataSchema {
    pub name: String,
    pub version: String,
}

impl From<DataSchema> for CoreDataSchema {
    fn from(value: DataSchema) -> Self {
        Self {
            name: value.name,
            version: value.version,
        }
    }
}

/// One typed recording operation emitted by `metric`.
#[napi(object)]
pub struct MetricMeasurement {
    pub name: String,
    pub kind: MetricKind,
    pub value_type: MetricValueType,
    #[napi(ts_type = "number")]
    pub value: Json,
    pub unit: Option<String>,
    pub description: Option<String>,
    pub attributes: Option<Json>,
    pub boundaries: Option<Vec<f64>>,
}

impl From<MetricMeasurement> for CoreMetricMeasurement {
    fn from(value: MetricMeasurement) -> Self {
        Self {
            name: value.name,
            kind: value.kind.into(),
            value_type: value.value_type.into(),
            value: value.value,
            unit: value.unit,
            description: value.description,
            attributes: value.attributes,
            boundaries: value.boundaries,
        }
    }
}

/// Canonical application-visible result of tool execution.
///
/// `result` is application-owned JSON. `annotation` is optional opaque adjacent
/// metadata that Relay transports without interpretation.
#[napi(object)]
pub struct ToolExecutionResult {
    pub result: Json,
    pub annotation: Option<Json>,
}

impl From<ToolExecutionResult> for CoreToolExecutionResult {
    fn from(value: ToolExecutionResult) -> Self {
        Self {
            result: value.result,
            annotation: value.annotation.filter(|annotation| !annotation.is_null()),
        }
    }
}

impl From<CoreToolExecutionResult> for ToolExecutionResult {
    fn from(value: CoreToolExecutionResult) -> Self {
        Self {
            result: value.result,
            annotation: value.annotation,
        }
    }
}

/// Per-call context delivered to a tool execution intercept.
///
/// `toolCallId` is the provider-issued correlation identifier recorded on
/// the managed tool call, or `null` when the call did not record one. It lets
/// an intercept that completes execution without invoking the remaining chain
/// associate its result with the originating tool call.
#[allow(dead_code, reason = "declared for generated TypeScript only")]
#[napi(object, use_nullable = true)]
pub struct ToolExecutionContext {
    pub tool_name: String,
    pub args: Json,
    pub tool_call_id: Option<String>,
}

/// A mark Relay materializes under a managed lifecycle.
#[allow(dead_code, reason = "declared for generated TypeScript only")]
#[napi(object)]
pub struct PendingMarkSpec {
    pub name: String,
    #[napi(ts_type = "string | null")]
    pub category: Option<String>,
    pub category_profile: Option<Json>,
    pub data: Option<Json>,
    #[napi(ts_type = "{ name: string; version: string } | null")]
    pub data_schema: Option<Json>,
    pub metadata: Option<Json>,
    #[napi(ts_type = "'trace' | 'debug' | 'info' | 'warn' | 'warning' | 'error' | null")]
    pub severity: Option<String>,
}

/// Handle to an isolated scope stack for per-request/per-task isolation.
#[napi]
pub struct ScopeStack {
    pub(crate) inner: ScopeStackHandle,
    pub(crate) publication_buffer: Option<PublicationBuffer>,
}

#[napi]
impl ScopeStack {
    /// Creates a new isolated scope stack with its own root scope.
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: create_scope_stack(),
            publication_buffer: None,
        }
    }
}

impl From<ScopeStackHandle> for ScopeStack {
    fn from(h: ScopeStackHandle) -> Self {
        Self {
            inner: h,
            publication_buffer: None,
        }
    }
}

/// A handle to an execution scope in the agent runtime.
///
/// Scopes form a hierarchical stack representing the current execution context
/// (e.g., agent -> function -> tool). Use this handle to reference a specific scope
/// when pushing child scopes, emitting events, or making tool/LLM calls.
#[napi]
pub struct ScopeHandle {
    pub(crate) inner: CoreScopeHandle,
}

#[napi]
impl ScopeHandle {
    /// The unique identifier for this scope.
    #[napi(getter, catch_unwind)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// The human-readable name of this scope.
    #[napi(getter, catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// The type of this scope (Agent, Tool, Llm, etc.).
    #[napi(getter, catch_unwind)]
    pub fn scope_type(&self) -> ScopeType {
        self.inner.scope_type.into()
    }

    /// Bitfield of scope attributes (e.g., PARALLEL, RELOCATABLE).
    #[napi(getter, catch_unwind)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// The UUID of this scope's parent, or `null` if this is the root scope.
    #[napi(getter, catch_unwind)]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }

    /// Optional user-defined data associated with this scope.
    #[napi(getter, catch_unwind)]
    pub fn data(&self) -> Option<serde_json::Value> {
        self.inner.data.clone()
    }

    /// Optional metadata associated with this scope.
    #[napi(getter, catch_unwind)]
    pub fn metadata(&self) -> Option<serde_json::Value> {
        self.inner.metadata.clone()
    }
}

impl From<CoreScopeHandle> for ScopeHandle {
    fn from(h: CoreScopeHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle representing an in-progress tool call.
///
/// Returned by `toolCall()` and used to signal completion via `toolCallEnd()`.
#[napi]
pub struct ToolHandle {
    pub(crate) inner: CoreToolHandle,
}

#[napi]
impl ToolHandle {
    /// The unique identifier for this tool call.
    #[napi(getter, catch_unwind)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// The name of the tool being called.
    #[napi(getter, catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Bitfield of tool attributes (e.g., LOCAL).
    #[napi(getter, catch_unwind)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// The UUID of the parent scope that initiated this tool call, or `null`.
    #[napi(getter, catch_unwind)]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
}

impl From<CoreToolHandle> for ToolHandle {
    fn from(h: CoreToolHandle) -> Self {
        Self { inner: h }
    }
}

/// A handle representing an in-progress LLM call.
///
/// Returned by `llmCall()` and used to signal completion via `llmCallEnd()`.
#[napi]
pub struct LlmHandle {
    pub(crate) inner: CoreLlmHandle,
}

#[napi]
impl LlmHandle {
    /// The unique identifier for this LLM call.
    #[napi(getter, catch_unwind)]
    pub fn uuid(&self) -> String {
        self.inner.uuid.to_string()
    }

    /// The name of the LLM provider being called.
    #[napi(getter, catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Bitfield of LLM attributes (e.g., STATELESS, STREAMING).
    #[napi(getter, catch_unwind)]
    pub fn attributes(&self) -> u32 {
        self.inner.attributes.bits()
    }

    /// The UUID of the parent scope that initiated this LLM call, or `null`.
    #[napi(getter, catch_unwind)]
    pub fn parent_uuid(&self) -> Option<String> {
        self.inner.parent_uuid.map(|u| u.to_string())
    }
}

impl From<CoreLlmHandle> for LlmHandle {
    fn from(h: CoreLlmHandle) -> Self {
        Self { inner: h }
    }
}

/// An LLM request, encapsulating headers and content.
///
/// Construct via `new LlmRequest(headers, content)`.
#[napi]
pub struct LlmRequest {
    pub(crate) inner: CoreLlmRequest,
}

#[napi]
impl LlmRequest {
    /// Create a new LLM request from headers and content.
    #[napi(constructor, catch_unwind)]
    pub fn new(headers: serde_json::Value, content: serde_json::Value) -> Self {
        let headers = match headers {
            Json::Object(m) => m,
            _ => serde_json::Map::new(),
        };
        Self {
            inner: CoreLlmRequest { headers, content },
        }
    }

    /// The metadata headers as a JSON object.
    #[napi(getter, catch_unwind)]
    pub fn headers(&self) -> serde_json::Value {
        Json::Object(self.inner.headers.clone())
    }

    /// The request payload as a JSON value.
    #[napi(getter, catch_unwind)]
    pub fn content(&self) -> serde_json::Value {
        self.inner.content.clone()
    }
}

/// Observability fields returned by mark and scope event sanitizers.
#[napi(object)]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EventSanitizeFields {
    pub data: Option<Json>,
    pub category_profile: Option<Json>,
    pub metadata: Option<Json>,
}

pub(crate) fn event_sanitize_fields_from_json(
    value: Json,
) -> serde_json::Result<EventSanitizeFields> {
    let Some(object) = value.as_object() else {
        return Err(serde::de::Error::custom(
            "event sanitizer must return an object",
        ));
    };
    if !["data", "categoryProfile", "metadata"]
        .iter()
        .any(|field| object.contains_key(*field))
    {
        return Err(serde::de::Error::custom(
            "event sanitizer must return at least one sanitizer field",
        ));
    }
    serde_json::from_value(value)
}

fn decode_request(codec: &dyn LlmCodec, request: Json) -> Result<Json> {
    let request: CoreLlmRequest = serde_json::from_value(request)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    let annotated = codec.decode(&request)?;
    serde_json::to_value(&annotated).map_err(Error::from)
}

fn encode_request(codec: &dyn LlmCodec, annotated: Json, original: Json) -> Result<Json> {
    let annotated: AnnotatedLlmRequest = serde_json::from_value(annotated)
        .map_err(|e| Error::invalid_argument(format!("invalid AnnotatedLlmRequest: {e}")))?;
    let original: CoreLlmRequest = serde_json::from_value(original)
        .map_err(|e| Error::invalid_argument(format!("invalid LlmRequest: {e}")))?;
    let encoded = codec.encode(&annotated, &original)?;
    serde_json::to_value(&encoded).map_err(Error::from)
}

fn decode_response(codec: &dyn LlmResponseCodec, response: &Json) -> Result<Json> {
    let annotated = codec.decode_response(response)?;
    serde_json::to_value(&annotated).map_err(Error::from)
}

/// Built-in codec for the OpenAI Chat Completions API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new OpenAIChatCodec()`.
#[napi(js_name = "OpenAIChatCodec")]
pub struct OpenAIChatCodec {
    codec: nemo_relay_codecs::openai_chat::OpenAIChatCodec,
    response_codec: Arc<dyn LlmResponseCodec>,
}

#[napi]
impl OpenAIChatCodec {
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            codec: nemo_relay_codecs::openai_chat::OpenAIChatCodec,
            response_codec: nemo_relay_runtime::codec::response_codec(ProviderSurface::OpenAIChat),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi(catch_unwind)]
    pub fn decode(&self, request: Json) -> Result<Json> {
        decode_request(&self.codec, request)
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi(catch_unwind)]
    pub fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        encode_request(&self.codec, annotated, original)
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse", catch_unwind)]
    pub fn decode_response(&self, response: Json) -> Result<Json> {
        decode_response(&*self.response_codec, &response)
    }
}

/// Built-in codec for the OpenAI Responses API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new OpenAIResponsesCodec()`.
#[napi(js_name = "OpenAIResponsesCodec")]
pub struct OpenAIResponsesCodec {
    codec: nemo_relay_codecs::openai_responses::OpenAIResponsesCodec,
    response_codec: Arc<dyn LlmResponseCodec>,
}

#[napi]
impl OpenAIResponsesCodec {
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            codec: nemo_relay_codecs::openai_responses::OpenAIResponsesCodec,
            response_codec: nemo_relay_runtime::codec::response_codec(
                ProviderSurface::OpenAIResponses,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi(catch_unwind)]
    pub fn decode(&self, request: Json) -> Result<Json> {
        decode_request(&self.codec, request)
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi(catch_unwind)]
    pub fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        encode_request(&self.codec, annotated, original)
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse", catch_unwind)]
    pub fn decode_response(&self, response: Json) -> Result<Json> {
        decode_response(&*self.response_codec, &response)
    }
}

/// Built-in codec for the Gemini generateContent API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new GeminiGenerateContentCodec()`.
#[napi(js_name = "GeminiGenerateContentCodec")]
pub struct GeminiGenerateContentCodec {
    codec: nemo_relay_codecs::gemini_generate_content::GeminiGenerateContentCodec,
    response_codec: Arc<dyn LlmResponseCodec>,
}

#[napi]
impl GeminiGenerateContentCodec {
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            codec: nemo_relay_codecs::gemini_generate_content::GeminiGenerateContentCodec,
            response_codec: nemo_relay_runtime::codec::response_codec(
                ProviderSurface::GeminiGenerateContent,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi(catch_unwind)]
    pub fn decode(&self, request: Json) -> Result<Json> {
        decode_request(&self.codec, request)
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi(catch_unwind)]
    pub fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        encode_request(&self.codec, annotated, original)
    }

    /// Decode a raw LLM response into structured form.
    /// Throws a TypeError if the response contains a BigInt value.
    #[napi(js_name = "decodeResponse", catch_unwind)]
    pub fn decode_response(
        &self,
        env: Env,
        #[napi(ts_arg_type = "any")] response: Unknown,
    ) -> Result<Json> {
        decode_response(
            &*self.response_codec,
            &crate::convert::checked_json(&env, response)?,
        )
    }
}

/// Built-in codec for the Anthropic Messages API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new AnthropicMessagesCodec()`.
#[napi]
pub struct AnthropicMessagesCodec {
    codec: nemo_relay_codecs::anthropic::AnthropicMessagesCodec,
    response_codec: Arc<dyn LlmResponseCodec>,
}

#[napi]
impl AnthropicMessagesCodec {
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            codec: nemo_relay_codecs::anthropic::AnthropicMessagesCodec,
            response_codec: nemo_relay_runtime::codec::response_codec(
                ProviderSurface::AnthropicMessages,
            ),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi(catch_unwind)]
    pub fn decode(&self, request: Json) -> Result<Json> {
        decode_request(&self.codec, request)
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi(catch_unwind)]
    pub fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        encode_request(&self.codec, annotated, original)
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse", catch_unwind)]
    pub fn decode_response(&self, response: Json) -> Result<Json> {
        decode_response(&*self.response_codec, &response)
    }
}

/// Built-in codec for the OCI Generative AI chat API.
///
/// Implements both request codec (decode/encode) and response codec
/// (decodeResponse). Construct with `new OCIGenAIChatCodec()`.
#[napi(js_name = "OCIGenAIChatCodec")]
pub struct OCIGenAIChatCodec {
    codec: nemo_relay_codecs::oci_genai::OCIGenAIChatCodec,
    response_codec: Arc<dyn LlmResponseCodec>,
}

#[napi]
impl OCIGenAIChatCodec {
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            codec: nemo_relay_codecs::oci_genai::OCIGenAIChatCodec,
            response_codec: nemo_relay_runtime::codec::response_codec(ProviderSurface::OCIGenAI),
        }
    }

    /// Decode an opaque LLM request into structured form.
    #[napi(catch_unwind)]
    pub fn decode(&self, request: Json) -> Result<Json> {
        decode_request(&self.codec, request)
    }

    /// Encode structured changes back into an opaque LLM request.
    #[napi(catch_unwind)]
    pub fn encode(&self, annotated: Json, original: Json) -> Result<Json> {
        encode_request(&self.codec, annotated, original)
    }

    /// Decode a raw LLM response into structured form.
    #[napi(js_name = "decodeResponse", catch_unwind)]
    pub fn decode_response(&self, response: Json) -> Result<Json> {
        decode_response(&*self.response_codec, &response)
    }
}
