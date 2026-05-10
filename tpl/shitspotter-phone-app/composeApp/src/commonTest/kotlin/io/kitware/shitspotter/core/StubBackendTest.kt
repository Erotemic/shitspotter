package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

class StubDetectorBackendTest {

    private fun frame(w: Int, h: Int): FrameSource = object : FrameSource {
        override val width = w
        override val height = h
        override val rotationDegrees = 0
        override fun toRgb888(): ByteArray = ByteArray(w * h * 3)
    }

    @Test
    fun returns_one_box_per_call() {
        val b = StubDetectorBackend()
        val r = b.analyze(frame(640, 480))
        assertEquals(1, r.detections.size)
    }

    @Test
    fun box_stays_inside_frame_bounds() {
        val b = StubDetectorBackend()
        val w = 320
        val h = 240
        // Run many frames so the jitter sweeps the whole range; assert
        // every box stays inside.
        repeat(200) {
            val det = b.analyze(frame(w, h)).detections.first()
            val box = det.box
            assertTrue(box.left in 0f..w.toFloat(), "left out of bounds: ${box.left}")
            assertTrue(box.top in 0f..h.toFloat(), "top out of bounds: ${box.top}")
            assertTrue(box.right <= w.toFloat() + 0.01f, "right out: ${box.right}")
            assertTrue(box.bottom <= h.toFloat() + 0.01f, "bottom out: ${box.bottom}")
        }
    }

    @Test
    fun warmup_is_safe() {
        val b = StubDetectorBackend()
        b.warmup() // should not throw
    }

    @Test
    fun close_then_analyze_throws() {
        val b = StubDetectorBackend()
        b.close()
        assertFails { b.analyze(frame(1, 1)) }
    }

    @Test
    fun spec_id_round_trips_through_constructor() {
        val custom = ModelSpec.STUB.copy(modelId = "custom-stub")
        val b = StubDetectorBackend(custom)
        assertEquals("custom-stub", b.spec.modelId)
    }

    @Test
    fun timing_fields_are_finite_and_non_negative() {
        val r = StubDetectorBackend().analyze(frame(64, 64))
        assertTrue(r.preprocessMs >= 0.0)
        assertTrue(r.inferenceMs >= 0.0)
        assertTrue(r.postprocessMs >= 0.0)
        assertTrue(!r.preprocessMs.isNaN() && !r.inferenceMs.isNaN())
    }
}
