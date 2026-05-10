package io.kitware.shitspotter.core

/**
 * Shared CPU-side image preprocessing. Backends may bypass this for
 * hardware-accelerated paths, but having a reference implementation in
 * commonMain means the desktop harness and Android stub take the same
 * path, which makes regression testing far easier.
 *
 * Input is RGB888 (interleaved, top-left origin), output is FloatArray
 * laid out per [ModelSpec.inputLayout] (NCHW or NHWC) using the spec's
 * normalization.
 */
object Preprocessing {

    fun letterboxRgb(
        rgb: ByteArray,
        srcW: Int,
        srcH: Int,
        dstW: Int,
        dstH: Int,
        padValue: Int = 114,
    ): Pair<ByteArray, LetterboxParams> {
        require(rgb.size == srcW * srcH * 3) {
            "rgb buffer size ${rgb.size} != ${srcW * srcH * 3}"
        }
        val params = LetterboxParams.compute(srcW, srcH, dstW, dstH)
        val newW = (srcW * params.scale).toInt().coerceAtLeast(1)
        val newH = (srcH * params.scale).toInt().coerceAtLeast(1)
        val out = ByteArray(dstW * dstH * 3) { padValue.toByte() }

        // Nearest-neighbor for speed; backends may swap in bilinear.
        val padX = params.padX.toInt()
        val padY = params.padY.toInt()
        for (y in 0 until newH) {
            val srcY = (y / params.scale).toInt().coerceIn(0, srcH - 1)
            val outY = padY + y
            if (outY < 0 || outY >= dstH) continue
            for (x in 0 until newW) {
                val srcX = (x / params.scale).toInt().coerceIn(0, srcW - 1)
                val outX = padX + x
                if (outX < 0 || outX >= dstW) continue
                val srcOff = (srcY * srcW + srcX) * 3
                val dstOff = (outY * dstW + outX) * 3
                out[dstOff] = rgb[srcOff]
                out[dstOff + 1] = rgb[srcOff + 1]
                out[dstOff + 2] = rgb[srcOff + 2]
            }
        }
        return out to params
    }

    fun toFloatTensor(
        rgb: ByteArray,
        width: Int,
        height: Int,
        layout: InputLayout,
        colorOrder: ColorOrder,
        normalization: Normalization,
    ): FloatArray {
        require(rgb.size == width * height * 3) {
            "rgb buffer size ${rgb.size} != ${width * height * 3}"
        }
        val total = width * height * 3
        val out = FloatArray(total)
        val mean = normalization.mean
        val std = normalization.std
        val scale = normalization.scale
        val swapRb = colorOrder == ColorOrder.BGR
        when (layout) {
            InputLayout.NHWC -> {
                var i = 0
                while (i < total) {
                    val r = (rgb[i].toInt() and 0xFF) * scale
                    val g = (rgb[i + 1].toInt() and 0xFF) * scale
                    val b = (rgb[i + 2].toInt() and 0xFF) * scale
                    if (swapRb) {
                        out[i] = (b - mean[0]) / std[0]
                        out[i + 1] = (g - mean[1]) / std[1]
                        out[i + 2] = (r - mean[2]) / std[2]
                    } else {
                        out[i] = (r - mean[0]) / std[0]
                        out[i + 1] = (g - mean[1]) / std[1]
                        out[i + 2] = (b - mean[2]) / std[2]
                    }
                    i += 3
                }
            }
            InputLayout.NCHW -> {
                val plane = width * height
                var px = 0
                var i = 0
                while (i < total) {
                    val r = (rgb[i].toInt() and 0xFF) * scale
                    val g = (rgb[i + 1].toInt() and 0xFF) * scale
                    val b = (rgb[i + 2].toInt() and 0xFF) * scale
                    val c0 = if (swapRb) b else r
                    val c2 = if (swapRb) r else b
                    out[px] = (c0 - mean[0]) / std[0]
                    out[plane + px] = (g - mean[1]) / std[1]
                    out[2 * plane + px] = (c2 - mean[2]) / std[2]
                    px++
                    i += 3
                }
            }
        }
        return out
    }
}
