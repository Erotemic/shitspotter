package io.github.erotemic.shitspotter.desktop

import io.github.erotemic.shitspotter.core.AppState
import io.github.erotemic.shitspotter.core.FrameSource
import io.github.erotemic.shitspotter.core.StubDetectorBackend
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Drives DesktopHarness.runOnce against a synthetic frame and asserts
 * that the shared AppState ends up populated with the stub detector's
 * single fake box plus a finite FrameTelemetry record.
 */
class DesktopHarnessTest {

    private fun synthFrame(w: Int = 320, h: Int = 240): FrameSource = object : FrameSource {
        override val width = w
        override val height = h
        override val rotationDegrees = 0
        override fun toRgb888(): ByteArray = ByteArray(w * h * 3)
    }

    @Test
    fun runOnce_populates_state_with_stub_detection() {
        val state = AppState()
        val backend = StubDetectorBackend()
        val harness = DesktopHarness(state) { backend }

        // Sanity: pre-state has no detections.
        assertEquals(0, state.lastDetections.size)
        assertEquals(null, state.lastTelemetry)

        harness.runOnce(synthFrame())

        assertEquals(1, state.lastDetections.size)
        assertNotNull(state.lastTelemetry)
        assertEquals(320, state.lastFrameWidth)
        assertEquals(240, state.lastFrameHeight)
        // Telemetry fields are non-negative + finite.
        val tele = state.lastTelemetry!!
        assertTrue(tele.preprocessMs >= 0.0)
        assertTrue(tele.inferenceMs >= 0.0)
        assertTrue(tele.postprocessMs >= 0.0)
        assertTrue(tele.fpsRecent >= 0.0)
        assertEquals("stub-fake-detector", tele.modelId)
        assertEquals("stub-1.0", tele.runtimeBackend)
    }

    @Test
    fun runOnce_with_high_threshold_keeps_zero_detections() {
        val state = AppState().also { it.scoreThreshold = 1.0f }
        val backend = StubDetectorBackend()
        DesktopHarness(state) { backend }.runOnce(synthFrame())

        // The stub returns score in [0.1, 0.9]; threshold=1.0 rejects all.
        assertEquals(0, state.lastDetections.size)
    }

    @Test
    fun runOnce_picks_up_active_threshold_on_each_call() {
        val state = AppState()
        val backend = StubDetectorBackend()
        val harness = DesktopHarness(state) { backend }

        // Permissive threshold first → 1 detection.
        state.scoreThreshold = 0.0f
        harness.runOnce(synthFrame())
        assertEquals(1, state.lastDetections.size)

        // Then strict threshold → 0 detections on the next frame.
        state.scoreThreshold = 1.0f
        harness.runOnce(synthFrame())
        assertEquals(0, state.lastDetections.size)
    }
}
