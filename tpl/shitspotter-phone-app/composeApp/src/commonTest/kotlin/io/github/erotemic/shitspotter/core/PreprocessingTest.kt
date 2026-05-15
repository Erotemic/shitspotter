package io.github.erotemic.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PreprocessingTest {
    @Test
    fun letterbox_pad_color_default_is_114() {
        val src = ByteArray(2 * 2 * 3) { 255.toByte() } // 2x2 white
        val (out, params) = Preprocessing.letterboxRgb(src, 2, 2, 4, 4)
        assertEquals(4 * 4 * 3, out.size)
        assertTrue(params.padX >= 0f)
        assertTrue(params.padY >= 0f)
        // At a corner of the output, expect pad color (114).
        val cornerOff = (0 * 4 + 0) * 3
        // Skip if padding is exactly zero in this direction; default 4x4 from 2x2 means scale=2 → no pad.
        // Use a non-square dst to force padding.
        val (out2, params2) = Preprocessing.letterboxRgb(src, 2, 2, 4, 6)
        assertTrue(params2.padY > 0f)
        assertEquals(114, out2[0].toInt() and 0xFF)
    }

    @Test
    fun nchw_layout_separates_channels() {
        val w = 2; val h = 2
        // pixel0=R, pixel1=G, pixel2=B, pixel3=W
        val rgb = byteArrayOf(
            255.toByte(), 0, 0,
            0, 255.toByte(), 0,
            0, 0, 255.toByte(),
            255.toByte(), 255.toByte(), 255.toByte(),
        )
        val out = Preprocessing.toFloatTensor(rgb, w, h, InputLayout.NCHW, ColorOrder.RGB, Normalization(scale = 1f / 255f))
        assertEquals(w * h * 3, out.size)
        // Channel 0 (R) at pixel 0 = 1.0
        assertEquals(1.0f, out[0])
        // Channel 1 (G) at pixel 1 = 1.0  -> index = w*h + 1 = 5
        assertEquals(1.0f, out[w * h + 1])
        // Channel 2 (B) at pixel 2 = 1.0 -> index = 2*w*h + 2 = 10
        assertEquals(1.0f, out[2 * w * h + 2])
    }

    @Test
    fun nhwc_layout_keeps_interleaved() {
        val rgb = byteArrayOf(255.toByte(), 0, 0)
        val out = Preprocessing.toFloatTensor(rgb, 1, 1, InputLayout.NHWC, ColorOrder.RGB, Normalization(scale = 1f / 255f))
        assertEquals(3, out.size)
        assertEquals(1.0f, out[0])
        assertEquals(0.0f, out[1])
        assertEquals(0.0f, out[2])
    }

    @Test
    fun bgr_swaps_red_and_blue() {
        val rgb = byteArrayOf(255.toByte(), 0, 0)
        val out = Preprocessing.toFloatTensor(rgb, 1, 1, InputLayout.NHWC, ColorOrder.BGR, Normalization(scale = 1f / 255f))
        assertEquals(0.0f, out[0]) // was R, now B
        assertEquals(0.0f, out[1])
        assertEquals(1.0f, out[2]) // was B, now R
    }
}
