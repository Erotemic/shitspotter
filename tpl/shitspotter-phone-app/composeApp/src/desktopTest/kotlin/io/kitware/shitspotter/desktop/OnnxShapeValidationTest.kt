package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.ModelSpec
import java.io.File
import kotlin.test.Test
import kotlin.test.assertFails

/**
 * Companion to OnnxBackendSmokeTest. Skips itself if the YOLOX-nano poop
 * ONNX file is not on disk. When the file is present, this test verifies
 * that constructing a backend with a deliberately wrong ModelSpec
 * (640x640 stub spec) fails loudly with a shape-mismatch error rather
 * than blowing up later inside session.run.
 */
class OnnxShapeValidationTest {

    private fun candidateModelPath(): File? {
        val candidates = listOf(
            File("../poop_models/yolox_nano_poop_cropped_only_best.onnx"),
            File("../../tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx"),
            File("/home/joncrall/code/shitspotter/tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx"),
        )
        return candidates.firstOrNull { it.isFile && it.length() > 0 }
    }

    @Test
    fun mismatched_input_shape_throws_at_init() {
        val model = candidateModelPath() ?: run {
            println("[shape] no YOLOX model on disk — skipping")
            return
        }
        // Stub spec is 640x640; the YOLOX-nano model is 416x416. The
        // backend should reject the spec at construction time, not at
        // first analyze().
        val wrongSpec = ModelSpec.STUB.copy(
            modelId = "wrong-shape-test",
            format = io.kitware.shitspotter.core.ModelFormat.ONNX,
            inputWidth = 640,
            inputHeight = 640,
        )
        assertFails {
            OnnxRuntimeJvmBackend(wrongSpec, model.absolutePath)
        }
    }
}
