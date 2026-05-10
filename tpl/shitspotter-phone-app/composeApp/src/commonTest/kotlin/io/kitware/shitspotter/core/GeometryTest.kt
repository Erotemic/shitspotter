package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GeometryTest {
    @Test
    fun bbox_intersect_when_overlapping() {
        val a = BoundingBox(0f, 0f, 10f, 10f)
        val b = BoundingBox(5f, 5f, 10f, 10f)
        val i = a.intersect(b)!!
        assertEquals(5f, i.x)
        assertEquals(5f, i.y)
        assertEquals(5f, i.width)
        assertEquals(5f, i.height)
    }

    @Test
    fun bbox_no_intersect_when_disjoint() {
        val a = BoundingBox(0f, 0f, 10f, 10f)
        val b = BoundingBox(20f, 20f, 5f, 5f)
        assertNull(a.intersect(b))
    }

    @Test
    fun iou_self_is_one() {
        val a = BoundingBox(1f, 2f, 4f, 6f)
        assertEquals(1f, a.iou(a))
    }

    @Test
    fun iou_disjoint_is_zero() {
        val a = BoundingBox(0f, 0f, 1f, 1f)
        val b = BoundingBox(10f, 10f, 1f, 1f)
        assertEquals(0f, a.iou(b))
    }

    @Test
    fun nms_keeps_best_and_drops_overlap() {
        val a = Detection(BoundingBox(0f, 0f, 10f, 10f), score = 0.9f, classId = 0)
        val b = Detection(BoundingBox(1f, 1f, 10f, 10f), score = 0.8f, classId = 0)
        val c = Detection(BoundingBox(50f, 50f, 5f, 5f), score = 0.7f, classId = 0)
        val out = Nms.apply(listOf(a, b, c), iouThreshold = 0.3f)
        assertEquals(2, out.size)
        assertTrue(out.contains(a))
        assertTrue(out.contains(c))
    }

    @Test
    fun letterbox_centers_square() {
        val p = LetterboxParams.compute(640, 480, 416, 416)
        assertEquals(416f / 640f, p.scale, "scale = min(dst/src)")
        assertEquals(0f, p.padX)
        assertTrue(p.padY > 0f)
    }

    @Test
    fun letterbox_round_trip() {
        val p = LetterboxParams.compute(640, 480, 416, 416)
        // Box at center of source ≈ box at center of letterboxed image.
        val srcCx = 320f
        val srcCy = 240f
        val w = 60f
        val h = 40f
        val srcBox = BoundingBox(srcCx - w / 2f, srcCy - h / 2f, w, h)
        val mapped = BoundingBox(
            x = srcBox.x * p.scale + p.padX,
            y = srcBox.y * p.scale + p.padY,
            width = srcBox.width * p.scale,
            height = srcBox.height * p.scale,
        )
        val back = p.mapBoxToSource(mapped)
        // Tolerance because of float math.
        assertTrue((back.x - srcBox.x) < 1f)
        assertTrue((back.y - srcBox.y) < 1f)
        assertTrue((back.width - srcBox.width) < 1f)
        assertTrue((back.height - srcBox.height) < 1f)
    }

    @Test
    fun yolox_decoded_postprocess_filters_below_threshold() {
        val numClasses = 1
        val numAnchors = 3
        val stride = 5 + numClasses
        val preds = FloatArray(numAnchors * stride)
        // anchor 0: high score
        preds[0] = 100f; preds[1] = 100f; preds[2] = 50f; preds[3] = 50f; preds[4] = 0.9f; preds[5] = 0.9f
        // anchor 1: low score
        preds[6] = 200f; preds[7] = 200f; preds[8] = 50f; preds[9] = 50f; preds[10] = 0.1f; preds[11] = 0.1f
        // anchor 2: high score, but very close to anchor 0
        preds[12] = 105f; preds[13] = 102f; preds[14] = 50f; preds[15] = 50f; preds[16] = 0.85f; preds[17] = 0.9f
        val dets = Yolox.postprocessDecoded(preds, numAnchors, numClasses, 0.25f, 0.45f, listOf("poop"))
        assertEquals(1, dets.size)
        assertEquals(0, dets[0].classId)
    }
}
