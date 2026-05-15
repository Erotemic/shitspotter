package io.github.erotemic.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Verifies that decodeRawStrides → postprocessDecoded round-trips a
 * synthetic raw output through to detections without losing the
 * argmax class or the box geometry. Catches sigmoid/exp errors in the
 * decode path that the unit tests in YoloxRawStridesTest could miss.
 */
class YoloxRoundTripTest {

    @Test
    fun raw_decode_then_postprocess_recovers_synthetic_box() {
        // Construct a single-stride YOLOX output where one anchor
        // unambiguously beats the threshold.
        val numClasses = 1
        val perRow = 5 + numClasses
        val grids = listOf(2 to 2)  // 2x2 grid
        val strides = listOf(8)
        val anchors = 4
        val raw = FloatArray(anchors * perRow)

        // Anchor index 1 (yi=0, xi=1) is the high-confidence one.
        // Its raw box is delta=0 → cx = (0+1)*8 = 8, cy = (0+0)*8 = 0,
        // raw w = log(2.0) ≈ 0.693 → exp(0.693)*8 = 16, h same.
        val high = 1 * perRow
        raw[high + 0] = 0f
        raw[high + 1] = 0f
        raw[high + 2] = kotlin.math.ln(2.0f)
        raw[high + 3] = kotlin.math.ln(2.0f)
        raw[high + 4] = 5f   // sigmoid(5) ≈ 0.993
        raw[high + 5] = 5f   // sigmoid(5) ≈ 0.993

        val decoded = Yolox.decodeRawStrides(raw, grids, strides, numClasses)

        val dets = Yolox.postprocessDecoded(
            decoded, anchors, numClasses,
            scoreThreshold = 0.5f,
            iouThreshold = 0.45f,
            classNames = listOf("poop"),
        )
        assertEquals(1, dets.size)
        val d = dets.first()
        // Score should be > 0.95 from sigmoid(5)^2 ≈ 0.987
        assertTrue(d.score > 0.95f)
        // Box centre at (cx=8, cy=0), size 16x16 → corner (0, -8, 16, 16)
        assertTrue(kotlin.math.abs(d.box.x - 0f) < 1f, "x=${d.box.x}")
        assertTrue(kotlin.math.abs(d.box.y - (-8f)) < 1f, "y=${d.box.y}")
        assertTrue(kotlin.math.abs(d.box.width - 16f) < 1f, "w=${d.box.width}")
        assertTrue(kotlin.math.abs(d.box.height - 16f) < 1f, "h=${d.box.height}")
    }

    @Test
    fun decode_then_postprocess_threshold_filters() {
        val numClasses = 1
        val perRow = 5 + numClasses
        val grids = listOf(1 to 1)
        val strides = listOf(8)
        val raw = FloatArray(perRow)
        // Score will be sigmoid(0)^2 = 0.25; should be filtered at 0.5.
        raw[0] = 0f; raw[1] = 0f
        raw[2] = 0f; raw[3] = 0f
        raw[4] = 0f; raw[5] = 0f
        val decoded = Yolox.decodeRawStrides(raw, grids, strides, numClasses)
        val dets = Yolox.postprocessDecoded(
            decoded, 1, numClasses, 0.5f, 0.45f, listOf("poop"),
        )
        assertEquals(0, dets.size)
    }
}
