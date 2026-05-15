package io.github.erotemic.shitspotter.desktop

import io.github.erotemic.shitspotter.core.FrameSource
import io.github.erotemic.shitspotter.core.ModelSpec
import java.io.File
import kotlin.test.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Conditional smoke test for the JVM ONNX backend. Skips itself (passes
 * without exercising anything) if no model file is reachable, so the
 * test suite still runs cleanly in environments where the
 * `tpl/poop_models/` submodule has not been pulled.
 *
 * When a model file is present, this test does NOT assert on detection
 * counts (the synthetic frame won't contain poop) — it only asserts
 * that the model loads, inference runs, and the output schema is
 * compatible with our YOLOX postprocess.
 */
class OnnxBackendSmokeTest {

    private fun candidateModelPath(): File? {
        val candidates = listOf(
            File("../poop_models/yolox_nano_poop_cropped_only_best.onnx"),
            File("../../tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx"),
            File("/home/joncrall/code/shitspotter/tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx"),
        )
        return candidates.firstOrNull { it.isFile && it.length() > 0 }
    }

    @Test
    fun yolox_nano_loads_and_runs_or_is_skipped() {
        val model = candidateModelPath() ?: run {
            println("[smoke] no YOLOX model on disk — skipping")
            return
        }
        println("[smoke] loading $model (${model.length() / 1024} KiB)")
        val spec = ModelSpec.YOLOX_NANO_POOP
        val backend = OnnxRuntimeJvmBackend(spec, model.absolutePath)

        // Synthetic 480x640 frame, all-grey to avoid hitting model-format quirks.
        val frame = object : FrameSource {
            override val width: Int = 640
            override val height: Int = 480
            override val rotationDegrees: Int = 0
            override fun toRgb888(): ByteArray = ByteArray(width * height * 3) { 0x80.toByte() }
        }

        val result = backend.analyze(frame)
        assertNotNull(result)
        // A dog/grey frame should not produce false positives at the default
        // 0.25 score threshold — but if the model is fundamentally a bbox
        // detector and threshold is right, the size of detections is low.
        // We don't assert exact count; we assert sanity: never NaN, sane
        // timings, no exceptions.
        assertTrue(result.inferenceMs >= 0.0)
        assertTrue(result.preprocessMs >= 0.0)
        assertTrue(result.postprocessMs >= 0.0)
        println(
            "[smoke] inference ok: dets=${result.detections.size} " +
                "pre=${"%.2f".format(result.preprocessMs)}ms " +
                "inf=${"%.2f".format(result.inferenceMs)}ms " +
                "post=${"%.2f".format(result.postprocessMs)}ms",
        )
        backend.close()
    }
}
