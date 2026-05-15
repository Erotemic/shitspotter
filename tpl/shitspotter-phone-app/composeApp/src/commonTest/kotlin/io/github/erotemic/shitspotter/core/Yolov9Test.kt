package io.github.erotemic.shitspotter.core

import kotlin.math.abs
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

class Yolov9DecodeTest {

    private fun logitFor(prob: Float): Float = ln(prob / (1f - prob))

    /** Build a single-class, single-level grid where one specific cell
     *  has a high-confidence detection and every other cell is silent. */
    private fun makeLevel(
        gridH: Int,
        gridW: Int,
        stride: Int,
        hotYi: Int,
        hotXi: Int,
        hotProb: Float = 0.95f,
        hotLTRB: FloatArray = floatArrayOf(2f, 2f, 2f, 2f),
    ): FeatureLevel {
        val hw = gridH * gridW
        val cls = FloatArray(hw) { -10f }   // sigmoid(-10) ≈ 4.5e-5
        cls[hotYi * gridW + hotXi] = logitFor(hotProb)
        val bbox = FloatArray(4 * hw)        // all-zero distances = degenerate
        for (k in 0 until 4) {
            bbox[k * hw + (hotYi * gridW + hotXi)] = hotLTRB[k]
        }
        return FeatureLevel(
            classLogits = cls,
            bbox = bbox,
            gridH = gridH,
            gridW = gridW,
            stride = stride,
        )
    }

    @Test
    fun decode_recovers_single_hot_cell() {
        // 2x2 grid at stride 8. Hot cell at (yi=1, xi=0) with prob ≈ 0.95
        // and stride-units distances (2,2,2,2) → pixel-units (16,16,16,16).
        // cx = (0+0.5)*8 = 4, cy = (1+0.5)*8 = 12.
        // x1 = 4-16=-12, y1=12-16=-4, x2=4+16=20, y2=12+16=28.
        val level = makeLevel(2, 2, 8, hotYi = 1, hotXi = 0)
        val dets = Yolov9.decode(
            levels = listOf(level),
            numClasses = 1,
            scoreThreshold = 0.5f,
            iouThreshold = 0.45f,
            classNames = listOf("poop"),
        )
        assertEquals(1, dets.size)
        val d = dets.first()
        assertTrue(d.score > 0.9f, "score=${d.score}")
        assertEquals(-12f, d.box.x)
        assertEquals(-4f, d.box.y)
        assertEquals(32f, d.box.width)   // 20 - (-12)
        assertEquals(32f, d.box.height)  // 28 - (-4)
        assertEquals(0, d.classId)
        assertEquals("poop", d.className)
    }

    @Test
    fun decode_filters_below_threshold() {
        val level = makeLevel(2, 2, 8, hotYi = 0, hotXi = 0, hotProb = 0.3f)
        val dets = Yolov9.decode(
            levels = listOf(level),
            numClasses = 1,
            scoreThreshold = 0.5f,
            iouThreshold = 0.45f,
            classNames = listOf("poop"),
        )
        assertEquals(0, dets.size)
    }

    @Test
    fun decode_filters_degenerate_zero_size_box() {
        // Hot cell with zero distances on every side → degenerate
        // (x1==x2 / y1==y2). Decoder should drop it before NMS.
        val level = makeLevel(
            gridH = 2, gridW = 2, stride = 8,
            hotYi = 0, hotXi = 0,
            hotLTRB = floatArrayOf(0f, 0f, 0f, 0f),
        )
        val dets = Yolov9.decode(
            levels = listOf(level),
            numClasses = 1,
            scoreThreshold = 0.5f,
            iouThreshold = 0.45f,
            classNames = listOf("poop"),
        )
        assertEquals(0, dets.size)
    }

    @Test
    fun decode_collapses_overlap_across_levels() {
        // Two levels (stride 8 and 16) with hot cells covering nearly
        // the same image region — same centre and similar size, so the
        // IoU is high. iouThreshold=0.2 (< their actual IoU ~0.25) lets
        // NMS collapse to just the higher-score one.
        val s8 = makeLevel(
            gridH = 4, gridW = 4, stride = 8,
            hotYi = 1, hotXi = 1,    // pixel center (12, 12)
            hotProb = 0.7f,
            hotLTRB = floatArrayOf(1f, 1f, 1f, 1f),  // → box (4..20, 4..20)
        )
        val s16 = makeLevel(
            gridH = 2, gridW = 2, stride = 16,
            hotYi = 0, hotXi = 0,    // pixel center (8, 8)
            hotProb = 0.9f,
            hotLTRB = floatArrayOf(1f, 1f, 1f, 1f),  // → box (-8..24, -8..24)
        )
        val dets = Yolov9.decode(
            levels = listOf(s8, s16),
            numClasses = 1,
            scoreThreshold = 0.5f,
            iouThreshold = 0.2f,
            classNames = listOf("poop"),
        )
        assertEquals(1, dets.size)
        assertTrue(abs(dets[0].score - 0.9f) < 0.02f)
    }

    @Test
    fun decode_multi_class_picks_argmax() {
        // 1x1 grid, 3 classes; class 2 wins.
        val hw = 1
        val classLogits = FloatArray(3 * hw)
        classLogits[0 * hw] = logitFor(0.2f)
        classLogits[1 * hw] = logitFor(0.5f)
        classLogits[2 * hw] = logitFor(0.95f)
        val bbox = floatArrayOf(2f, 2f, 2f, 2f)  // one cell, 4 distances
        val level = FeatureLevel(
            classLogits = classLogits,
            bbox = bbox,
            gridH = 1, gridW = 1, stride = 8,
        )
        val dets = Yolov9.decode(
            levels = listOf(level),
            numClasses = 3,
            scoreThreshold = 0.5f,
            iouThreshold = 0.45f,
            classNames = listOf("a", "b", "c"),
        )
        assertEquals(1, dets.size)
        assertEquals(2, dets[0].classId)
        assertEquals("c", dets[0].className)
    }

    @Test
    fun decode_bbox_already_in_pixels_skips_stride_multiply() {
        // Same hot cell as decode_recovers_single_hot_cell but with
        // bboxAlreadyInPixels=true so the distances are taken verbatim.
        val level = makeLevel(2, 2, 8, hotYi = 1, hotXi = 0).copy(
            bboxAlreadyInPixels = true,
        )
        val dets = Yolov9.decode(
            levels = listOf(level),
            numClasses = 1,
            scoreThreshold = 0.5f,
            iouThreshold = 0.45f,
            classNames = listOf("poop"),
        )
        assertEquals(1, dets.size)
        // cx=4, cy=12; distances 2 each (not multiplied by 8):
        // box = (4-2, 12-2, 4, 4) = (2, 10, 4, 4)
        val d = dets.first()
        assertEquals(2f, d.box.x)
        assertEquals(10f, d.box.y)
        assertEquals(4f, d.box.width)
        assertEquals(4f, d.box.height)
    }

    @Test
    fun decode_throws_on_wrong_classLogits_size() {
        val level = FeatureLevel(
            classLogits = FloatArray(3),  // expected 1 * 2 * 2 = 4
            bbox = FloatArray(16),
            gridH = 2, gridW = 2, stride = 8,
        )
        assertFails {
            Yolov9.decode(listOf(level), 1, 0.5f, 0.45f, listOf("poop"))
        }
    }

    @Test
    fun decode_throws_on_wrong_bbox_size() {
        val level = FeatureLevel(
            classLogits = FloatArray(4),
            bbox = FloatArray(7),  // expected 4 * 2 * 2 = 16
            gridH = 2, gridW = 2, stride = 8,
        )
        assertFails {
            Yolov9.decode(listOf(level), 1, 0.5f, 0.45f, listOf("poop"))
        }
    }
}
