package io.kitware.shitspotter.core

/**
 * Pure-Kotlin reference implementation of YOLOX postprocessing for the
 * specific export format we ship with: a single tensor of shape
 * [1, num_anchors, 4 + 1 + num_classes] where the 4 are
 * (cx, cy, w, h) in **input-image pixel space** that has already been
 * decoded against strides 8 / 16 / 32. This is the typical YOLOX-nano
 * exported-with-decode shape used in mobile demos.
 *
 * The function does NOT do letterbox un-mapping itself — that is the
 * caller's job via [LetterboxParams.mapBoxToSource] so that desktop and
 * Android share the same code path.
 *
 * If the loaded ONNX model turns out to expose the *raw* per-stride
 * outputs instead, the backend's adapter must concatenate strides and
 * call [decodeRawStrides] before this function.
 */
object Yolox {

    /**
     * Decode predictions where the model already emitted absolute
     * (cx, cy, w, h, obj, cls0, cls1, ...) per anchor in input space.
     */
    fun postprocessDecoded(
        predictions: FloatArray,
        numAnchors: Int,
        numClasses: Int,
        scoreThreshold: Float,
        iouThreshold: Float,
        classNames: List<String>,
    ): List<Detection> {
        val stride = 5 + numClasses
        require(predictions.size >= numAnchors * stride) {
            "predictions buffer too small: ${predictions.size} < ${numAnchors * stride}"
        }
        val out = ArrayList<Detection>(numAnchors / 8)
        var p = 0
        for (i in 0 until numAnchors) {
            val cx = predictions[p]
            val cy = predictions[p + 1]
            val w = predictions[p + 2]
            val h = predictions[p + 3]
            val obj = predictions[p + 4]
            var bestCls = 0
            var bestClsScore = -1f
            for (c in 0 until numClasses) {
                val s = predictions[p + 5 + c]
                if (s > bestClsScore) {
                    bestClsScore = s
                    bestCls = c
                }
            }
            val score = obj * bestClsScore
            if (score >= scoreThreshold && w > 0f && h > 0f) {
                val box = BoundingBox(cx - w / 2f, cy - h / 2f, w, h)
                out += Detection(
                    box = box,
                    score = score,
                    classId = bestCls,
                    className = classNames.getOrNull(bestCls),
                )
            }
            p += stride
        }
        return Nms.apply(out, iouThreshold)
    }

    /**
     * Decode raw multi-stride YOLOX output (cx, cy, w, h are stride-relative;
     * objectness and class scores are sigmoid-pre logits). [strides] gives
     * the anchor stride per row in [predictions]; total rows must match
     * `predictions.size / (5 + numClasses)`.
     *
     * This is provided so that the eventual real ONNX model can be plugged
     * in regardless of whether its export embeds the decode step.
     */
    fun decodeRawStrides(
        predictions: FloatArray,
        gridSizes: List<Pair<Int, Int>>, // (h, w) per stride in same order as strides
        strides: List<Int>,
        numClasses: Int,
    ): FloatArray {
        require(gridSizes.size == strides.size) {
            "gridSizes and strides must align"
        }
        val perRow = 5 + numClasses
        val totalAnchors = gridSizes.sumOf { (h, w) -> h * w }
        require(predictions.size == totalAnchors * perRow) {
            "predictions size ${predictions.size} does not match expected $totalAnchors x $perRow"
        }
        val decoded = FloatArray(predictions.size)
        var rowIdx = 0
        for (i in gridSizes.indices) {
            val (gh, gw) = gridSizes[i]
            val s = strides[i].toFloat()
            for (yi in 0 until gh) {
                for (xi in 0 until gw) {
                    val base = rowIdx * perRow
                    val cx = (predictions[base] + xi) * s
                    val cy = (predictions[base + 1] + yi) * s
                    val w = kotlin.math.exp(predictions[base + 2]) * s
                    val h = kotlin.math.exp(predictions[base + 3]) * s
                    decoded[base] = cx
                    decoded[base + 1] = cy
                    decoded[base + 2] = w
                    decoded[base + 3] = h
                    decoded[base + 4] = sigmoid(predictions[base + 4])
                    for (c in 0 until numClasses) {
                        decoded[base + 5 + c] = sigmoid(predictions[base + 5 + c])
                    }
                    rowIdx++
                }
            }
        }
        return decoded
    }

    private fun sigmoid(x: Float): Float = (1f / (1f + kotlin.math.exp(-x)))
}
