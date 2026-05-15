package io.github.erotemic.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class FpsCounterTest {
    @Test
    fun fewer_than_two_marks_returns_zero() {
        val c = FpsCounter()
        assertEquals(0.0, c.mark(0L))
    }

    @Test
    fun ten_evenly_spaced_marks_at_100ms_yields_10_fps() {
        val c = FpsCounter(windowMs = 1500L)
        var t = 0L
        repeat(10) { c.mark(t); t += 100L }
        // 10 marks → 9 intervals of 100 ms = 900 ms → 10 fps within the 1500 ms window
        val fps = c.mark(t)
        assertTrue(fps in 9.0..11.0, "expected ~10 fps, got $fps")
    }

    @Test
    fun marks_outside_window_are_dropped() {
        // Window is 200 ms. We seed two old marks that are all > 200 ms in
        // the past relative to the third, then add a fourth mark close to
        // the third — the old two are dropped, only [t3, t4] remain, which
        // is enough to yield a non-zero FPS.
        val c = FpsCounter(windowMs = 200L)
        c.mark(0L)
        c.mark(100L)
        c.mark(450L)  // both 0 and 100 fall outside [250, 450]; window now [450]
        val fps = c.mark(550L)  // window now [450, 550]
        // 1 interval of 100 ms across 2 marks → ~10 fps
        assertTrue(fps in 5.0..15.0, "expected ~10 fps after stale window prune, got $fps")
    }
}

class LatencyAccumulatorTest {
    @Test
    fun mean_handles_empty() {
        assertEquals(0.0, LatencyAccumulator().mean())
    }

    @Test
    fun mean_matches_simple_average() {
        val acc = LatencyAccumulator()
        acc.record(10.0)
        acc.record(20.0)
        acc.record(30.0)
        assertEquals(20.0, acc.mean())
    }

    @Test
    fun window_size_caps_history() {
        val acc = LatencyAccumulator(windowSize = 3)
        repeat(5) { acc.record(it.toDouble()) }
        // Only last 3 (2.0, 3.0, 4.0) should contribute.
        assertEquals(3.0, acc.mean())
    }

    @Test
    fun percentile_works_for_simple_set() {
        val acc = LatencyAccumulator(windowSize = 100)
        for (i in 1..100) acc.record(i.toDouble())
        assertTrue(acc.percentile(0.5) in 49.0..51.0)
        assertTrue(acc.percentile(0.9) in 89.0..91.0)
    }
}

class ModelRegistryTest {
    @Test
    fun default_resolves() {
        val s = ModelRegistry.default
        assertEquals("stub-fake-detector", s.modelId)
    }

    @Test
    fun by_id_finds_yolox() {
        val s = ModelRegistry.byId("yolox-nano-poop-cropped-v1")
        assertTrue(s != null)
        assertEquals(416, s!!.inputWidth)
        assertEquals(416, s.inputHeight)
        assertEquals(InputLayout.NCHW, s.inputLayout)
        assertEquals(ColorOrder.RGB, s.colorOrder)
        assertEquals(PostprocessType.YOLOX, s.postprocessType)
    }

    @Test
    fun by_id_returns_null_for_missing() {
        assertEquals(null, ModelRegistry.byId("nope"))
    }

    @Test
    fun all_models_have_unique_ids() {
        val ids = ModelRegistry.all.map { it.modelId }
        assertEquals(ids.size, ids.toSet().size)
    }
}

class AppStateTest {
    @Test
    fun pushFrame_updates_state() {
        val s = AppState()
        val det = Detection(BoundingBox(0f, 0f, 10f, 10f), score = 0.9f)
        val tele = sampleTelemetry()
        s.pushFrame(listOf(det), tele, 320, 240)
        assertEquals(1, s.lastDetections.size)
        assertEquals(tele, s.lastTelemetry)
        assertEquals(320, s.lastFrameWidth)
        assertEquals(240, s.lastFrameHeight)
    }

    @Test
    fun setError_then_pushFrame_clears_error() {
        val s = AppState()
        s.setError("boom")
        assertEquals("boom", s.lastError)
        s.pushFrame(emptyList(), sampleTelemetry(), 1, 1)
        assertEquals(null, s.lastError)
    }

    @Test
    fun activeModel_falls_back_to_default_when_unknown() {
        val s = AppState()
        s.activeModelId = "made-up"
        assertEquals(ModelRegistry.default.modelId, s.activeModel().modelId)
    }

    private fun sampleTelemetry(): FrameTelemetry = FrameTelemetry(
        deviceModel = "test", osVersion = "test", appCommit = "test",
        modelId = "stub", modelHash = null,
        runtimeBackend = "stub", delegate = null,
        inputWidth = 1, inputHeight = 1,
        captureMs = 0.0, preprocessMs = 0.0, inferenceMs = 0.0,
        postprocessMs = 0.0, overlayMs = 0.0, fpsRecent = 0.0,
        detectionCount = 0, droppedFrames = 0L,
    )
}
