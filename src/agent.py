"""
LiveKit Voice AI Agent with Multi-Agent RAG System
==================================================

This module implements a voice-enabled AI agent that:
- Accepts voice input via Deepgram STT
- Routes queries through a multi-agent system (RAG, Tools, General)
- Generates voice responses via Cartesia TTS
- Logs comprehensive performance metrics

Architecture:
    Voice Input → STT → Multi-Agent System → TTS → Voice Output
    
Performance:
    - STT latency: ~0.5s
    - Agent processing: 1-3s
    - TTS generation: ~0.3s
    - Total end-to-end: 2-5s
"""

import logging
import threading
import signal
import contextlib
from multi_agent_rag import graph  # 🎯 Import multi-agent RAG system
from metrics_logger import get_metrics_logger  # 📊 Metrics tracking
from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
)
from livekit.agents import metrics as lk_metrics  # LiveKit metrics
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero, langchain
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Monkey patch to fix signal handling in non-main thread on Windows
try:
    import livekit.agents.ipc.supervised_proc as sp
    
    @contextlib.contextmanager
    def _safe_mask_ctrl_c():
        """Safe SIGINT masking that only runs in main thread"""
        if threading.current_thread() is threading.main_thread():
            old = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                yield
            finally:
                signal.signal(signal.SIGINT, old)
        else:
            # Non-main thread: skip signal masking
            yield
    
    sp._mask_ctrl_c = _safe_mask_ctrl_c
except (ImportError, AttributeError):
    # If the module structure changes, fail gracefully
    pass

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    """
    Voice AI Assistant with Multi-Agent Capabilities.
    
    This assistant provides:
    - Document-based Q&A through RAG (5 comprehensive documents)
    - Real-time information via API tools (weather, currency, time)
    - Natural conversation handling
    
    The assistant is optimized for voice interaction with:
    - Concise responses (2-3 sentences by default)
    - Source citations for factual claims
    - Conversational tone
    
    Attributes:
        instructions: System prompt defining assistant capabilities
    """
    
    def __init__(self) -> None:
        """
        Initialize the Assistant with instructions and capabilities.
        """
        super().__init__(
            instructions="""You are a helpful AI assistant with access to a comprehensive knowledge base and real-time tools.

**Knowledge Base Topics:**
- Artificial Intelligence and Machine Learning
- Climate Change and Environmental Science
- Modern World History (1900-present)
- Blockchain Technology and Cryptocurrencies
- Health, Wellness, and Medicine

**Real-Time Tools:**
- Weather information for any city
- Currency exchange rates
- Current time in different timezones

**Guidelines:**
- Provide accurate, concise information optimized for voice interaction
- Cite sources when using the knowledge base
- Use tools when requesting current information
- Be conversational, friendly, and helpful
- Keep responses brief (2-3 sentences) unless more detail is requested
- No complex formatting, emojis, or symbols in responses""",
        )

    # Optional: Add any specific function tools here if needed
    # The multi-agent system handles tool calling internally
    @function_tool
    async def sample_function(self, context: RunContext, param: str):
        """
        Sample function tool (placeholder for future extensions).
        
        Args:
            param: Sample parameter
        """
        logger.info(f"Sample function called with: {param}")
        return "Sample function response"


def prewarm(proc: JobProcess):
    """
    Pre-warm models and initialize systems before agent starts.
    
    This function runs once during agent startup to load heavy resources
    into memory, reducing cold-start latency for first user interaction.
    
    Operations performed:
    1. Load VAD (Voice Activity Detection) model for turn detection
    2. Initialize RAG system (load FAISS vector store)
    3. Pre-load embeddings model
    
    Args:
        proc: JobProcess instance containing shared user data
        
    Side Effects:
        - Sets proc.userdata["vad"] with loaded VAD model
        - Sets proc.userdata["rag_ready"] flag indicating RAG availability
        - Sets proc.userdata["metrics"] with global metrics logger
    """
    logger.info("Starting pre-warm sequence...")
    
    # Initialize metrics logger
    metrics = get_metrics_logger(log_file="metrics/agent_metrics.json")
    proc.userdata["metrics"] = metrics
    logger.info("✓ Metrics logger initialized")
    
    # Load VAD model
    logger.info("Loading VAD model...")
    proc.userdata["vad"] = silero.VAD.load()
    
    # Initialize RAG system (load vector store)
    logger.info("Pre-warming RAG system...")
    try:
        from rag_system import get_rag_system
        rag = get_rag_system()
        proc.userdata["rag_ready"] = True
        logger.info("✅ RAG system pre-warmed successfully")
    except Exception as e:
        logger.error(f"⚠️ Warning: Could not pre-warm RAG system: {e}")
        proc.userdata["rag_ready"] = False


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for the LiveKit voice agent.
    
    This async function:
    1. Sets up the voice pipeline (STT, LLM, TTS)
    2. Configures turn detection and VAD
    3. Initializes metrics collection
    4. Connects to the LiveKit room
    5. Handles user voice interactions
    
    Args:
        ctx: JobContext containing room, process, and configuration
        
    Lifecycle:
        - Called when a new user joins the room
        - Runs until user disconnects or agent is stopped
        - Cleanup performed via shutdown callbacks
    """
    # Get metrics logger
    metrics = ctx.proc.userdata.get("metrics")
    if metrics:
        metrics.increment_query_count()
        logger.info(f"Session started (Query #{metrics.total_queries})")
    
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=langchain.LLMAdapter(
            graph=graph  # 🎬 Use your compiled LangGraph movie agent
        ),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = lk_metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """
        Handle metrics collection events from LiveKit pipeline.
        
        This callback receives metrics for:
        - STT latency (speech-to-text processing time)
        - LLM inference time
        - TTS generation time (text-to-speech)
        - Turn detection timing
        
        Args:
            ev: MetricsCollectedEvent containing pipeline metrics
        """
        lk_metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
        
        # Log to custom metrics logger
        custom_metrics = ctx.proc.userdata.get("metrics")
        if custom_metrics and hasattr(ev.metrics, 'ttft'):
            # TTFT: Time to First Token (STT + LLM start)
            if ev.metrics.ttft:
                custom_metrics.record_metric(
                    "ttft",
                    ev.metrics.ttft,
                    metadata={"pipeline": "voice"}
                )
            
            # Log STT, LLM, TTS separately if available
            if hasattr(ev.metrics, 'stt_latency'):
                custom_metrics.record_metric(
                    "stt_latency",
                    ev.metrics.stt_latency,
                    metadata={"provider": "deepgram", "model": "nova-3"}
                )
            
            if hasattr(ev.metrics, 'llm_latency'):
                custom_metrics.record_metric(
                    "llm_inference",
                    ev.metrics.llm_latency,
                    metadata={"provider": "langchain", "backend": "multi-agent"}
                )
            
            if hasattr(ev.metrics, 'tts_latency'):
                custom_metrics.record_metric(
                    "tts_generation",
                    ev.metrics.tts_latency,
                    metadata={"provider": "cartesia"}
                )

    async def log_usage():
        """
        Log usage summary on session shutdown.
        """
        summary = usage_collector.get_summary()
        logger.info(f"LiveKit Usage Summary: {summary}")
        
        # Log custom metrics summary
        custom_metrics = ctx.proc.userdata.get("metrics")
        if custom_metrics:
            custom_metrics.log_summary()
            custom_metrics.save_to_json()

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
