package io.github.erotemic.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PreprocessingResizeTest {

    @Test
    fun stretchRgb_produces_expected_size() {
        val src = ByteArray(2 * 2 * 3) { 255.toByte() }
        val (out, _) = Preprocessing.stretchRgb(src, 2, 2, 4, 6)
        assertEquals(4 * 6 * 3, out.size)
        // All-white in source → all-white in dst (within nearest-neighbour).
        out.forEach { assertEquals(255.toByte(), it) }
    }

    @Test
    fun stretchRgb_throws_on_size_mismatch() {
        var threw = false
        try {
            Preprocessing.stretchRgb(ByteArray(5), 2, 2, 4, 4)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue(threw)
    }

    @Test
    fun centerCropRgb_widens_to_target_aspect() {
        // 4x2 source (aspect=2.0), 4x4 dst (aspect=1.0) → crop sides
        val src = ByteArray(4 * 2 * 3) { 100.toByte() }
        val (out, params) = Preprocessing.centerCropRgb(src, 4, 2, 4, 4)
        assertEquals(4 * 4 * 3, out.size)
        assertTrue(params.padX < 0f, "centre-crop encodes the crop offset as negative pad")
    }

    @Test
    fun centerCropRgb_taller_to_target_aspect() {
        // 2x4 source (aspect=0.5), 4x4 dst (aspect=1.0) → crop top/bottom
        val src = ByteArray(2 * 4 * 3) { 100.toByte() }
        val (out, params) = Preprocessing.centerCropRgb(src, 2, 4, 4, 4)
        assertEquals(4 * 4 * 3, out.size)
        assertTrue(params.padY < 0f)
    }

    @Test
    fun centerCropRgb_throws_on_size_mismatch() {
        var threw = false
        try {
            Preprocessing.centerCropRgb(ByteArray(5), 2, 2, 4, 4)
        } catch (e: IllegalArgumentException) {
            threw = true
        }
        assertTrue(threw)
    }
}
