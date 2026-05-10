package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.BackendComparisonReport
import io.kitware.shitspotter.core.FailureCaseSerialization
import kotlinx.serialization.decodeFromString
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.io.path.createTempDirectory
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration test that drives the actual `CompareCli.run` against a
 * synthetic image and writes a JSON report. Catches regressions in
 * the entire CLI pipeline (arg parsing → frame load → backend
 * dispatch → JSON output) without requiring the optional ONNX model
 * file.
 */
class CompareCliEndToEndTest {

    private fun makeImageFile(): File {
        val dir = createTempDirectory("ssp-cli-e2e").toFile()
        val img = BufferedImage(64, 48, BufferedImage.TYPE_INT_RGB)
        for (y in 0 until 48) for (x in 0 until 64) {
            img.setRGB(x, y, (x * 4 + y) and 0xFFFFFF)
        }
        val f = File(dir, "test.jpg")
        ImageIO.write(img, "jpg", f)
        return f
    }

    @Test
    fun help_returns_zero() {
        val rc = CompareCli.run(arrayOf("--help"))
        assertEquals(0, rc)
    }

    @Test
    fun stub_only_run_writes_json_and_returns_zero() {
        val image = makeImageFile()
        val outFile = File(image.parentFile, "report.json")
        try {
            val rc = CompareCli.run(
                arrayOf(
                    "--image=${image.absolutePath}",
                    "--runs=1",
                    "--warmup=0",
                    "--out=${outFile.absolutePath}",
                ),
            )
            assertEquals(0, rc)
            assertTrue(outFile.isFile)
            val rep = FailureCaseSerialization.json.decodeFromString<BackendComparisonReport>(
                outFile.readText()
            )
            assertEquals(1, rep.rows.size)
            val stub = rep.rows.first()
            assertEquals("stub-fake-detector", stub.modelId)
            assertEquals("stub-1.0", stub.backendName)
            assertNotNull(rep.timestamp)
        } finally {
            image.parentFile.deleteRecursively()
        }
    }

    @Test
    fun no_image_argument_throws_with_helpful_message() {
        var threw = false
        try {
            CompareCli.run(arrayOf("--runs=1"))
        } catch (e: IllegalStateException) {
            threw = true
            // Sanity: error message should mention --image so the user
            // knows what they forgot.
            assertTrue(e.message?.contains("--image") == true, "message: ${e.message}")
        }
        assertTrue(threw)
    }

    @Test
    fun no_stub_with_no_models_writes_empty_table() {
        val image = makeImageFile()
        val outFile = File(image.parentFile, "report.json")
        try {
            val rc = CompareCli.run(
                arrayOf(
                    "--image=${image.absolutePath}",
                    "--runs=1",
                    "--warmup=0",
                    "--no-stub",
                    "--out=${outFile.absolutePath}",
                ),
            )
            assertEquals(0, rc)
            val rep = FailureCaseSerialization.json.decodeFromString<BackendComparisonReport>(
                outFile.readText()
            )
            assertEquals(0, rep.rows.size)
        } finally {
            image.parentFile.deleteRecursively()
        }
    }
}
