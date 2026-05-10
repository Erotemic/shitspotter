package io.kitware.shitspotter.core

import kotlinx.serialization.Serializable

@Serializable
data class BoundingBox(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
) {
    val left: Float get() = x
    val top: Float get() = y
    val right: Float get() = x + width
    val bottom: Float get() = y + height
    val centerX: Float get() = x + width / 2f
    val centerY: Float get() = y + height / 2f
    val area: Float get() = width * height

    fun intersect(other: BoundingBox): BoundingBox? {
        val l = maxOf(left, other.left)
        val t = maxOf(top, other.top)
        val r = minOf(right, other.right)
        val b = minOf(bottom, other.bottom)
        if (r <= l || b <= t) return null
        return BoundingBox(l, t, r - l, b - t)
    }

    fun iou(other: BoundingBox): Float {
        val inter = intersect(other) ?: return 0f
        val u = area + other.area - inter.area
        if (u <= 0f) return 0f
        return inter.area / u
    }
}

@Serializable
data class Detection(
    val box: BoundingBox,
    val score: Float,
    val classId: Int = 0,
    val className: String? = null,
)

object Nms {
    fun apply(
        detections: List<Detection>,
        iouThreshold: Float = 0.45f,
    ): List<Detection> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.score }.toMutableList()
        val keep = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val best = sorted.removeAt(0)
            keep += best
            sorted.removeAll { it.classId == best.classId && it.box.iou(best.box) > iouThreshold }
        }
        return keep
    }
}

data class LetterboxParams(
    val scale: Float,
    val padX: Float,
    val padY: Float,
    val outWidth: Int,
    val outHeight: Int,
    val srcWidth: Int,
    val srcHeight: Int,
) {
    fun mapBoxToSource(b: BoundingBox): BoundingBox {
        val x = (b.x - padX) / scale
        val y = (b.y - padY) / scale
        val w = b.width / scale
        val h = b.height / scale
        return BoundingBox(
            x = x.coerceIn(0f, srcWidth.toFloat()),
            y = y.coerceIn(0f, srcHeight.toFloat()),
            width = w.coerceAtMost(srcWidth - x).coerceAtLeast(0f),
            height = h.coerceAtMost(srcHeight - y).coerceAtLeast(0f),
        )
    }

    companion object {
        fun compute(srcW: Int, srcH: Int, dstW: Int, dstH: Int): LetterboxParams {
            val scale = minOf(dstW.toFloat() / srcW, dstH.toFloat() / srcH)
            val newW = srcW * scale
            val newH = srcH * scale
            val padX = (dstW - newW) / 2f
            val padY = (dstH - newH) / 2f
            return LetterboxParams(scale, padX, padY, dstW, dstH, srcW, srcH)
        }
    }
}
