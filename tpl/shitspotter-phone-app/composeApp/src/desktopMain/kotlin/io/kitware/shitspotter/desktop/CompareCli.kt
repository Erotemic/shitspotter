package io.kitware.shitspotter.desktop

import io.kitware.shitspotter.core.BackendComparison
import io.kitware.shitspotter.core.BackendComparisonReport
import io.kitware.shitspotter.core.BuildInfo
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FailureCaseSerialization
import io.kitware.shitspotter.core.ModelRegistry
import io.kitware.shitspotter.core.ModelSpec
import io.kitware.shitspotter.core.PrintlnLogger
import io.kitware.shitspotter.core.StubDetectorBackend
import kotlinx.datetime.Clock
import kotlinx.serialization.encodeToString
import java.io.File
import kotlin.system.exitProcess

/**
 * Headless backend-comparison CLI for the desktop target. Run with:
 *
 *   ./gradlew :composeApp:run --args="compare \
 *      --image=/path/to/test.jpg \
 *      --model=/path/to/yolox_nano_poop_cropped_only_best.onnx \
 *      --runs=5 --warmup=1 \
 *      --out=compare_report.json"
 *
 * Output is a markdown-style table on stdout plus an optional JSON
 * report at --out, suitable for archiving next to a benchmark.
 */
object CompareCli {

    fun run(args: Array<String>): Int {
        val image = argValue(args, "--image")?.let { File(it) }
            ?: error("compare needs --image=<path>")
        require(image.isFile) { "image not found: ${image.absolutePath}" }
        val modelPath = argValue(args, "--model")?.let { File(it) }
        // Default to the YOLOX-nano poop spec when an ONNX path is given but
        // no model-id, since the stub spec's 640x640 input shape will not
        // match a 416x416 YOLOX model.
        val modelId = argValue(args, "--model-id") ?: if (modelPath != null) {
            ModelSpec.YOLOX_NANO_POOP.modelId
        } else {
            ModelRegistry.default.modelId
        }
        val runs = argValue(args, "--runs")?.toIntOrNull() ?: 5
        val warmup = argValue(args, "--warmup")?.toIntOrNull() ?: 1
        val outFile = argValue(args, "--out")?.let { File(it) }
        val thresholdOverride = argValue(args, "--score-threshold")?.toFloatOrNull()

        val frame = StillImageFrameSource.fromFile(image)
        val backends = mutableListOf<DetectorBackend>()
        backends += StubDetectorBackend()  // sanity baseline

        if (modelPath != null && modelPath.isFile) {
            val baseSpec: ModelSpec = ModelRegistry.byId(modelId) ?: ModelRegistry.default
            val spec = if (thresholdOverride != null) baseSpec.copy(scoreThreshold = thresholdOverride) else baseSpec
            try {
                backends += OnnxRuntimeJvmBackend(spec, modelPath.absolutePath)
            } catch (t: Throwable) {
                PrintlnLogger.error("ShitSpotter.Compare", "could not load $modelPath", t)
            }
        }

        val rows = BackendComparison.runMeasured(
            frame = frame,
            backends = backends,
            warmupRuns = warmup,
            measureRuns = runs,
        )
        val report = BackendComparisonReport(
            timestamp = Clock.System.now().toString(),
            deviceModel = BuildInfo.deviceModel,
            osVersion = BuildInfo.osVersion,
            rows = rows,
        )

        println("# ShitSpotter backend comparison")
        println("# image:  ${image.absolutePath}")
        println("# device: ${report.deviceModel}")
        println("# os:     ${report.osVersion}")
        println("# runs:   $runs (warmup=$warmup)")
        println()
        println(BackendComparison.renderTable(rows))

        if (outFile != null) {
            outFile.writeText(FailureCaseSerialization.json.encodeToString(report))
            println()
            println("# wrote ${outFile.absolutePath}")
        }

        backends.forEach { runCatching { it.close() } }
        return 0
    }

    private fun argValue(args: Array<String>, key: String): String? {
        val prefix = "$key="
        args.forEach { if (it.startsWith(prefix)) return it.removePrefix(prefix) }
        return null
    }
}

/**
 * Stand-alone main entry point if you want to run the comparison without
 * spinning up the GUI. The Gradle desktop application stays pointed at
 * `Main.kt`, which dispatches to this when invoked with the literal
 * `compare` arg as args[0].
 */
fun runCompareIfRequested(args: Array<String>): Boolean {
    if (args.isEmpty() || args[0] != "compare") return false
    val rest = args.drop(1).toTypedArray()
    val rc = CompareCli.run(rest)
    if (rc != 0) exitProcess(rc)
    return true
}
