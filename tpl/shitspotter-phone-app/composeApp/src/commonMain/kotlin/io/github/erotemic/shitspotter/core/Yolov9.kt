package io.github.erotemic.shitspotter.core

import kotlin.math.exp

/**
 * Postprocess for YOLOv9-style anchor-free detectors that export each
 * feature level as a separate (class-logit, decoded-bbox) tensor pair.
 *
 * Expected per-level shape (NCHW, batch=1, flattened):
 *   classLogits: numClasses * H * W (raw logits — sigmoid applied here)
 *   bbox:        4 * H * W (left/top/right/bottom distance in
 *                stride-units, i.e. multiply by `stride` to get pixels)
 *
 * The "stride-units" assumption matches DFL-decoded exports from the
 * ultralytics YOLOv8/YOLOv9 pipeline with reg_max=16. If a future
 * export bakes the *stride into the bbox values, set
 * [FeatureLevel.bboxAlreadyInPixels] = true to skip the multiplication.
 *
 * Detections come out of [decode] already NMS'd at [iouThreshold].
 */
data class FeatureLevel(
    val classLogits: FloatArray,
    val bbox: FloatArray,
    val gridH: Int,
    val gridW: Int,
    val stride: Int,
    val bboxAlreadyInPixels: Boolean = false,
)

object Yolov9 {

    fun decode(
        levels: List<FeatureLevel>,
        numClasses: Int,
        scoreThreshold: Float,
        iouThreshold: Float,
        classNames: List<String>,
    ): List<Detection> {
        val all = mutableListOf<Detection>()
        for (level in levels) {
            decodeLevelInto(level, numClasses, scoreThreshold, classNames, all)
        }
        return Nms.apply(all, iouThreshold)
    }

    private fun decodeLevelInto(
        level: FeatureLevel,
        numClasses: Int,
        scoreThreshold: Float,
        classNames: List<String>,
        out: MutableList<Detection>,
    ) {
        require(level.classLogits.size == numClasses * level.gridH * level.gridW) {
            "classLogits size ${level.classLogits.size} != $numClasses*${level.gridH}*${level.gridW}"
        }
        require(level.bbox.size == 4 * level.gridH * level.gridW) {
            "bbox size ${level.bbox.size} != 4*${level.gridH}*${level.gridW}"
        }
        val hw = level.gridH * level.gridW
        val strideF = level.stride.toFloat()
        val bboxMult = if (level.bboxAlreadyInPixels) 1f else strideF
        for (yi in 0 until level.gridH) {
            for (xi in 0 until level.gridW) {
                val cellIdx = yi * level.gridW + xi
                var bestClass = 0
                var bestLogit = level.classLogits[cellIdx]
                for (c in 1 until numClasses) {
                    val logit = level.classLogits[c * hw + cellIdx]
                    if (logit > bestLogit) {
                        bestLogit = logit
                        bestClass = c
                    }
                }
                val score = sigmoid(bestLogit)
                if (score < scoreThreshold) continue
                val lDist = level.bbox[0 * hw + cellIdx] * bboxMult
                val tDist = level.bbox[1 * hw + cellIdx] * bboxMult
                val rDist = level.bbox[2 * hw + cellIdx] * bboxMult
                val bDist = level.bbox[3 * hw + cellIdx] * bboxMult
                val cx = (xi + 0.5f) * strideF
                val cy = (yi + 0.5f) * strideF
                val x1 = cx - lDist
                val y1 = cy - tDist
                val x2 = cx + rDist
                val y2 = cy + bDist
                if (x2 <= x1 || y2 <= y1) continue
                out += Detection(
                    box = BoundingBox(x1, y1, x2 - x1, y2 - y1),
                    score = score,
                    classId = bestClass,
                    className = classNames.getOrNull(bestClass),
                )
            }
        }
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + exp(-x)))
}
