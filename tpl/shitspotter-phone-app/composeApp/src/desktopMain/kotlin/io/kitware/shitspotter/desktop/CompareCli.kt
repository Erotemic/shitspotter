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
        if (args.any { it == "--help" || it == "-h" }) {
            printHelp()
            return 0
        }
        val image = argValue(args, "--image")?.let { File(it) }
            ?: error("compare needs --image=<path> (try --help)")
        require(image.isFile) { "image not found: ${image.absolutePath}" }
        // Multiple --model can be passed; --model-id at the same index
        // selects the ModelSpec. If model-ids are not given the loader
        // tries to infer from the filename.
        val modelArgs = args.filter { it.startsWith("--model=") }.map { it.removePrefix("--model=") }
        val modelIdArgs = args.filter { it.startsWith("--model-id=") }.map { it.removePrefix("--model-id=") }
        val runs = argValue(args, "--runs")?.toIntOrNull() ?: 5
        val warmup = argValue(args, "--warmup")?.toIntOrNull() ?: 1
        val outFile = argValue(args, "--out")?.let { File(it) }
        val thresholdOverride = argValue(args, "--score-threshold")?.toFloatOrNull()
        val noStub = args.any { it == "--no-stub" }

        val frame = StillImageFrameSource.fromFile(image)
        val backends = mutableListOf<DetectorBackend>()
        if (!noStub) backends += StubDetectorBackend()

        modelArgs.forEachIndexed { i, p ->
            val modelPath = File(p)
            if (!modelPath.isFile) {
                PrintlnLogger.warn("ShitSpotter.Compare", "skip missing model: $p")
                return@forEachIndexed
            }
            val explicitId = modelIdArgs.getOrNull(i)
            val inferredId = guessModelIdFromPath(modelPath.name)
            val modelId = explicitId ?: inferredId ?: ModelSpec.YOLOX_NANO_POOP.modelId
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

    /** Visible-for-test argument splitter. */
    internal fun argValue(args: Array<String>, key: String): String? {
        val prefix = "$key="
        args.forEach { if (it.startsWith(prefix)) return it.removePrefix(prefix) }
        return null
    }

    /** Visible-for-test repeated-argument collector. */
    internal fun argValues(args: Array<String>, key: String): List<String> =
        args.filter { it.startsWith("$key=") }.map { it.removePrefix("$key=") }

    private fun printHelp() {
        println(
            """
            usage:
              ./gradlew :composeApp:run --args="compare \\
                  --image=<path>                  # required
                  [--model=<onnx-path>]*          # repeat for multi-model compare
                  [--model-id=<id>]*              # match by index; ModelRegistry id
                  [--score-threshold=<float>]     # override ModelSpec default
                  [--runs=<int>]                  # measured runs per backend (default 5)
                  [--warmup=<int>]                # warmup runs before timing (default 1)
                  [--no-stub]                     # drop the stub baseline row
                  [--out=<json-path>]             # write a JSON report"

            known model ids:
              ${ModelRegistry.all.joinToString("\n              ") { "${it.modelId}: ${it.displayName}" }}

            related sub-commands:
              describe --model=<onnx-path>        # dump ONNX input/output shapes

            example:
              ./gradlew :composeApp:run --args="compare \\
                  --image=tpl/YOLOX/assets/dog.jpg \\
                  --model=tpl/poop_models/yolox_nano_poop_cropped_only_best.onnx \\
                  --runs=10 --warmup=2 --out=docs/benchmarks/desktop_dog.json"
            """.trimIndent(),
        )
    }

    /** Best-effort match of a known ModelRegistry id from a file name, so
     *  the caller can drop the bare ONNX path and still get the right
     *  preprocessing/postprocessing config. Visible for test. */
    internal fun guessModelIdFromPath(fileName: String): String? {
        val lower = fileName.lowercase()
        return ModelRegistry.all.firstOrNull { spec ->
            val mfLower = spec.modelFile.lowercase()
            mfLower.isNotBlank() && lower == mfLower
        }?.modelId
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

/**
 * Quick model introspection: dump the ONNX input/output names + shapes
 * + dtype to stdout. Useful when adding a new ModelSpec — confirm the
 * actual shape before guessing.
 *
 *   ./gradlew :composeApp:run --args="describe --model=/path/to.onnx"
 */
fun runDescribeIfRequested(args: Array<String>): Boolean {
    if (args.isEmpty() || args[0] != "describe") return false
    val modelPath = args.firstOrNull { it.startsWith("--model=") }?.removePrefix("--model=")
        ?: error("describe needs --model=<path>")
    val f = File(modelPath)
    require(f.isFile) { "model not found: ${f.absolutePath}" }
    val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
    val opts = ai.onnxruntime.OrtSession.SessionOptions()
    val s = env.createSession(f.absolutePath, opts)
    println("# describe ${f.absolutePath}")
    println()
    println("inputs:")
    for (name in s.inputNames) {
        val info = s.inputInfo[name]!!
        val ti = info.info as? ai.onnxruntime.TensorInfo
        if (ti != null) {
            println("  $name : shape=${ti.shape.toList()} type=${ti.type}")
        } else {
            println("  $name : (non-tensor)")
        }
    }
    println("outputs:")
    for (name in s.outputNames) {
        val info = s.outputInfo[name]!!
        val ti = info.info as? ai.onnxruntime.TensorInfo
        if (ti != null) {
            println("  $name : shape=${ti.shape.toList()} type=${ti.type}")
        } else {
            println("  $name : (non-tensor)")
        }
    }
    s.close()
    return true
}
