package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertTrue

class YoloxPostprocessExtraTest {

    @Test
    fun multi_class_picks_argmax() {
        // 1 anchor, 3 classes; class 1 is the highest at 0.9.
        val numClasses = 3
        val perRow = 5 + numClasses
        val preds = FloatArray(1 * perRow)
        preds[0] = 50f; preds[1] = 50f; preds[2] = 20f; preds[3] = 30f
        preds[4] = 0.9f                                      // obj
        preds[5] = 0.1f; preds[6] = 0.9f; preds[7] = 0.4f    // cls0/1/2
        val dets = Yolox.postprocessDecoded(
            preds, 1, numClasses,
            scoreThreshold = 0.25f, iouThreshold = 0.45f,
            classNames = listOf("a", "b", "c"),
        )
        assertEquals(1, dets.size)
        assertEquals(1, dets[0].classId)
        assertEquals("b", dets[0].className)
        // Score = obj * cls = 0.9 * 0.9 = 0.81
        assertTrue(dets[0].score in 0.80f..0.82f)
    }

    @Test
    fun zero_size_anchors_are_filtered() {
        val numClasses = 1
        val perRow = 5 + numClasses
        val preds = FloatArray(2 * perRow)
        // anchor 0: w=0 → filtered out even at high score
        preds[0] = 50f; preds[1] = 50f; preds[2] = 0f; preds[3] = 30f
        preds[4] = 0.9f; preds[5] = 0.9f
        // anchor 1: h=0 → filtered out
        preds[6] = 50f; preds[7] = 50f; preds[8] = 30f; preds[9] = 0f
        preds[10] = 0.9f; preds[11] = 0.9f
        val dets = Yolox.postprocessDecoded(
            preds, 2, numClasses, 0.25f, 0.45f, listOf("poop"),
        )
        assertEquals(0, dets.size)
    }

    @Test
    fun box_coordinates_are_corner_form() {
        // anchor centered at (50, 50) with width=20, height=10 should
        // produce a box (40, 45, 20, 10) in left-top-w-h form.
        val preds = floatArrayOf(50f, 50f, 20f, 10f, 0.9f, 0.9f)
        val dets = Yolox.postprocessDecoded(
            preds, 1, 1, 0.25f, 0.45f, listOf("poop"),
        )
        assertEquals(1, dets.size)
        val box = dets[0].box
        assertEquals(40f, box.x)
        assertEquals(45f, box.y)
        assertEquals(20f, box.width)
        assertEquals(10f, box.height)
    }

    @Test
    fun nms_collapses_overlapping_high_score_anchors() {
        val numClasses = 1
        val perRow = 5 + numClasses
        val preds = FloatArray(3 * perRow)
        // Three anchors over the same region with descending score.
        for (i in 0 until 3) {
            preds[i * perRow + 0] = 50f
            preds[i * perRow + 1] = 50f
            preds[i * perRow + 2] = 20f
            preds[i * perRow + 3] = 20f
            preds[i * perRow + 4] = 0.9f
            preds[i * perRow + 5] = 0.9f - 0.01f * i
        }
        val dets = Yolox.postprocessDecoded(
            preds, 3, numClasses, 0.25f, 0.45f, listOf("poop"),
        )
        // After NMS only the top-scoring anchor should remain.
        assertEquals(1, dets.size)
    }

    @Test
    fun input_buffer_too_small_throws() {
        assertFails {
            Yolox.postprocessDecoded(
                predictions = FloatArray(3),
                numAnchors = 5,
                numClasses = 1,
                scoreThreshold = 0.25f,
                iouThreshold = 0.45f,
                classNames = listOf("poop"),
            )
        }
    }
}
