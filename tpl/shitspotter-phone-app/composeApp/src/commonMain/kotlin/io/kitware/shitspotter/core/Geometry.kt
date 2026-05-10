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

    /**
     * Rotate this box clockwise by 0/90/180/270 degrees within a frame
     * of size [frameW] x [frameH]. Returns the new box and the new frame
     * dimensions as a pair: `(rotatedBox, newFrameWidth to newFrameHeight)`.
     *
     * The box is assumed to be in pixel space of the source frame. Other
     * rotation amounts throw — Android camera rotationDegrees is always
     * a multiple of 90.
     */
    fun rotated(degrees: Int, frameW: Int, frameH: Int): Triple<BoundingBox, Int, Int> {
        val d = ((degrees % 360) + 360) % 360
        return when (d) {
            0 -> Triple(this, frameW, frameH)
            90 -> Triple(
                BoundingBox(
                    x = frameH - (y + height),
                    y = x,
                    width = height,
                    height = width,
                ),
                frameH, frameW,
            )
            180 -> Triple(
                BoundingBox(
                    x = frameW - (x + width),
                    y = frameH - (y + height),
                    width = width,
                    height = height,
                ),
                frameW, frameH,
            )
            270 -> Triple(
                BoundingBox(
                    x = y,
                    y = frameW - (x + width),
                    width = height,
                    height = width,
                ),
                frameH, frameW,
            )
            else -> error("rotation must be a multiple of 90; got $degrees")
        }
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

/** Cheap post-backend filter so the UI can move the score threshold
 *  without re-running inference. */
fun List<Detection>.filterByScore(min: Float): List<Detection> =
    if (min <= 0f) this else filter { it.score >= min }

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
