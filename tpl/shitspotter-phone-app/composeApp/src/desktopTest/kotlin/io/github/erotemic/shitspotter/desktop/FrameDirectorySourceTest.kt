package io.github.erotemic.shitspotter.desktop

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class FrameDirectorySourceTest {

    private fun makeTempDirWithFrames(count: Int): File {
        val dir = createTempDirectory("ssp-frames").toFile()
        for (i in 0 until count) {
            val img = BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB)
            for (y in 0 until 8) for (x in 0 until 8) {
                img.setRGB(x, y, (i * 31 + x * 7 + y) and 0xFFFFFF)
            }
            ImageIO.write(img, "png", File(dir, "%04d.png".format(i)))
        }
        return dir
    }

    @Test
    fun reads_files_in_lexicographic_order() {
        val dir = makeTempDirWithFrames(3)
        try {
            val src = FrameDirectorySource(dir)
            assertEquals(3, src.frameCount)
            // Frames cycle and reset() rewinds the cursor.
            val a = src.next()
            val b = src.next()
            val c = src.next()
            val d = src.next()
            assertEquals(8, a.width)
            assertEquals(8, b.width)
            assertEquals(8, c.width)
            assertEquals(8, d.width)
            src.reset()
            val a2 = src.next()
            assertEquals(8, a2.width)
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun empty_directory_throws() {
        val dir = createTempDirectory("ssp-empty").toFile()
        try {
            assertFailsWith<IllegalArgumentException> { FrameDirectorySource(dir) }
        } finally {
            dir.deleteRecursively()
        }
    }

    @Test
    fun non_directory_path_throws() {
        val f = File.createTempFile("ssp-not-dir", ".txt")
        try {
            assertFailsWith<IllegalArgumentException> { FrameDirectorySource(f) }
        } finally {
            f.delete()
        }
    }
}
