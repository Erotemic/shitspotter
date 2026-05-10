package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.FrameSource
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/**
 * Loads a still image from disk and exposes it as a [FrameSource]. The
 * desktop harness uses this to feed the same shared pipeline that runs on
 * Android — same preprocessing, same postprocessing, same overlay math.
 */
class StillImageFrameSource(image: BufferedImage) : FrameSource {
    override val width: Int = image.width
    override val height: Int = image.height
    override val rotationDegrees: Int = 0

    private val rgb: ByteArray = run {
        // BufferedImage.getRGB returns ARGB ints. Convert once.
        val argb = IntArray(width * height)
        image.getRGB(0, 0, width, height, argb, 0, width)
        val out = ByteArray(width * height * 3)
        var dst = 0
        for (px in argb) {
            out[dst] = ((px ushr 16) and 0xFF).toByte()       // R
            out[dst + 1] = ((px ushr 8) and 0xFF).toByte()    // G
            out[dst + 2] = (px and 0xFF).toByte()             // B
            dst += 3
        }
        out
    }

    override fun toRgb888(): ByteArray = rgb.copyOf()

    companion object {
        fun fromFile(file: File): StillImageFrameSource {
            require(file.isFile) { "Not a file: ${file.absolutePath}" }
            val img = ImageIO.read(file) ?: error("Could not decode image: ${file.absolutePath}")
            return StillImageFrameSource(img)
        }
    }
}
