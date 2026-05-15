package io.github.erotemic.shitspotter.desktop

import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails

class StillImageFrameSourceTest {

    @Test
    fun reports_image_dimensions() {
        val img = BufferedImage(13, 7, BufferedImage.TYPE_INT_RGB)
        val src = StillImageFrameSource(img)
        assertEquals(13, src.width)
        assertEquals(7, src.height)
        assertEquals(0, src.rotationDegrees)
    }

    @Test
    fun toRgb888_size_matches_dimensions() {
        val img = BufferedImage(8, 4, BufferedImage.TYPE_INT_RGB)
        val src = StillImageFrameSource(img)
        assertEquals(8 * 4 * 3, src.toRgb888().size)
    }

    @Test
    fun toRgb888_returns_a_copy() {
        val img = BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB)
        val src = StillImageFrameSource(img)
        val a = src.toRgb888()
        val b = src.toRgb888()
        a[0] = 99
        // b should still be untouched.
        assertEquals(0, b[0].toInt())
    }

    @Test
    fun encodes_red_pixel_correctly() {
        val img = BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB)
        img.setRGB(0, 0, 0xFF0000)
        val rgb = StillImageFrameSource(img).toRgb888()
        assertEquals(255.toByte(), rgb[0])  // R
        assertEquals(0.toByte(), rgb[1])    // G
        assertEquals(0.toByte(), rgb[2])    // B
    }

    @Test
    fun fromFile_throws_on_directory() {
        val dir = createTempDirectory("ssp-still").toFile()
        try {
            assertFails { StillImageFrameSource.fromFile(dir) }
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun fromFile_loads_real_image() {
        val tmp = createTempDirectory("ssp-still").toFile()
        try {
            val img = BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB)
            for (y in 0 until 4) for (x in 0 until 4) img.setRGB(x, y, 0x00FF00)
            val f = java.io.File(tmp, "green.png")
            ImageIO.write(img, "png", f)
            val src = StillImageFrameSource.fromFile(f)
            assertEquals(4, src.width)
            assertEquals(4, src.height)
            val rgb = src.toRgb888()
            assertEquals(0.toByte(), rgb[0])     // R
            assertEquals(255.toByte(), rgb[1])   // G
            assertEquals(0.toByte(), rgb[2])     // B
        } finally {
            tmp.deleteRecursively()
        }
    }
}
