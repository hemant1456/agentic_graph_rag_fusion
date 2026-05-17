import time

try:
    import phoenix as px
    from opentelemetry import trace as otel_trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False

PHOENIX_PORT = 6006
OTEL_PROJECT = "agentic_rag_fusion"


class PhoenixExporter:
    """
    Manages a local Arize Phoenix server and the OTel tracer that feeds it.

    Design notes:
    - Phoenix runs in a background thread (run_in_thread=True) — it doesn't
      block the Python process.
    - Spans are exported synchronously via SimpleSpanProcessor so they appear
      in the UI immediately after each query completes.
    - Passing the returned tracer to TracedRAG is enough — no further
      configuration needed in the pipeline code.
    """

    def __init__(self, port: int = PHOENIX_PORT) -> None:
        if not PHOENIX_AVAILABLE:
            raise ImportError(
                "arize-phoenix / opentelemetry-sdk not installed.\n"
                "Run:  uv sync --extra step-02"
            )
        self.port = port
        self._session = None
        self._tracer = None

    @property
    def url(self) -> str:
        if self._session is not None:
            return self._session.url
        return f"http://localhost:{self.port}/"

    def start(self, startup_wait_s: float = 3.0) -> "otel_trace.Tracer":
        """
        Launch the Phoenix server and configure the OTel OTLP exporter.

        Args:
            startup_wait_s: seconds to wait before sending first spans
                            (server needs a moment to bind the port)

        Returns:
            A configured opentelemetry.trace.Tracer — pass this to TracedRAG
            as the `otel_tracer` argument.
        """
        import os
        os.environ.setdefault("PHOENIX_PORT", str(self.port))
        self._session = px.launch_app(run_in_thread=True)
        time.sleep(startup_wait_s)

        endpoint = f"http://localhost:{self.port}/v1/traces"
        exporter = OTLPSpanExporter(endpoint=endpoint)
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)

        self._tracer = otel_trace.get_tracer(OTEL_PROJECT)
        print(f"Phoenix running → {self.url}")
        return self._tracer
