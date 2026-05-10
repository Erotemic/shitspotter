package io.kitware.shitspotter.desktop

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.kitware.shitspotter.core.DetectorBackend
import io.kitware.shitspotter.core.FrameSource
import io.kitware.shitspotter.core.InferenceResult
import io.kitware.shitspotter.core.InputLayout
import io.kitware.shitspotter.core.LetterboxParams
import io.kitware.shitspotter.core.ModelSpec
import io.kitware.shitspotter.core.Nms
import io.kitware.shitspotter.core.PostprocessType
import io.kitware.shitspotter.core.Preprocessing
import io.kitware.shitspotter.core.ResizePolicy
import io.kitware.shitspotter.core.Yolox
import io.kitware.shitspotter.core.nowMonoMs
import java.io.File
import java.nio.FloatBuffer

/**
 * Desktop / JVM ONNX Runtime backend (CPU-only). Used by the still-image
 * harness so the same model + preprocessing + postprocessing as Android
 * can be exercised on Linux without a phone.
 */
class OnnxRuntimeJvmBackend(
    override val spec: ModelSpec,
    private val modelPath: String,
) : DetectorBackend {
    override val backendName: String = "onnxruntime-jvm-1.19"
    override val delegate: String = "CPU"

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private var closed = false

    init {
        require(File(modelPath).isFile) { "ONNX model missing: $modelPath" }
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        session = env.createSession(modelPath, opts)
        inputName = session.inputNames.first()
    }

    override fun warmup() {
        val dummy = ByteArray(spec.inputWidth * spec.inputHeight * 3)
        analyze(object : FrameSource {
            override val width = spec.inputWidth
            override val height = spec.inputHeight
            override val rotationDegrees = 0
            override fun toRgb888(): ByteArray = dummy
        })
    }

    override fun analyze(frame: FrameSource): InferenceResult {
        check(!closed) { "Backend already closed" }

        val preStart = nowMonoMs()
        val rgb = frame.toRgb888()
        val (lbBytes, lbParams) = if (spec.resizePolicy == ResizePolicy.LETTERBOX) {
            Preprocessing.letterboxRgb(rgb, frame.width, frame.height, spec.inputWidth, spec.inputHeight)
        } else {
            stretchRgb(rgb, frame.width, frame.height, spec.inputWidth, spec.inputHeight) to
                LetterboxParams.compute(frame.width, frame.height, spec.inputWidth, spec.inputHeight)
        }
        val tensorData = Preprocessing.toFloatTensor(
            rgb = lbBytes,
            width = spec.inputWidth,
            height = spec.inputHeight,
            layout = spec.inputLayout,
            colorOrder = spec.colorOrder,
            normalization = spec.normalization,
        )
        val shape: LongArray = when (spec.inputLayout) {
            InputLayout.NCHW -> longArrayOf(1L, 3L, spec.inputHeight.toLong(), spec.inputWidth.toLong())
            InputLayout.NHWC -> longArrayOf(1L, spec.inputHeight.toLong(), spec.inputWidth.toLong(), 3L)
        }
        val inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(tensorData), shape)
        val preprocessMs = nowMonoMs() - preStart

        val infStart = nowMonoMs()
        val outputs = session.run(mapOf(inputName to inputTensor))
        val inferenceMs = nowMonoMs() - infStart

        val postStart = nowMonoMs()
        val raw = outputs[0].value
        val flat = flattenOutput(raw)
        val numClasses = spec.classNames.size
        val perRow = 5 + numClasses
        require(flat.size % perRow == 0) {
            "ONNX output shape ${flat.size} not divisible by $perRow"
        }
        val numAnchors = flat.size / perRow
        val rawDets = when (spec.postprocessType) {
            PostprocessType.YOLOX,
            PostprocessType.YOLO_V9,
            PostprocessType.GENERIC_BOX_SCORE_CLASS -> Yolox.postprocessDecoded(
                predictions = flat,
                numAnchors = numAnchors,
                numClasses = numClasses,
                scoreThreshold = spec.scoreThreshold,
                iouThreshold = spec.iouThreshold,
                classNames = spec.classNames,
            )
            PostprocessType.STUB, PostprocessType.NONE -> emptyList()
        }
        val mappedDets = rawDets.map { it.copy(box = lbParams.mapBoxToSource(it.box)) }
        val finalDets = if (mappedDets.size > 1)
            Nms.apply(mappedDets, spec.iouThreshold) else mappedDets

        outputs.close()
        inputTensor.close()
        val postprocessMs = nowMonoMs() - postStart

        return InferenceResult(
            detections = finalDets,
            preprocessMs = preprocessMs,
            inferenceMs = inferenceMs,
            postprocessMs = postprocessMs,
            backendName = backendName,
            delegate = delegate,
        )
    }

    override fun close() {
        if (closed) return
        closed = true
        try { session.close() } catch (_: Throwable) {}
    }

    private fun stretchRgb(rgb: ByteArray, srcW: Int, srcH: Int, dstW: Int, dstH: Int): ByteArray {
        val out = ByteArray(dstW * dstH * 3)
        val sx = srcW.toFloat() / dstW
        val sy = srcH.toFloat() / dstH
        var dst = 0
        for (y in 0 until dstH) {
            val sYi = (y * sy).toInt().coerceIn(0, srcH - 1)
            for (x in 0 until dstW) {
                val sXi = (x * sx).toInt().coerceIn(0, srcW - 1)
                val srcOff = (sYi * srcW + sXi) * 3
                out[dst] = rgb[srcOff]
                out[dst + 1] = rgb[srcOff + 1]
                out[dst + 2] = rgb[srcOff + 2]
                dst += 3
            }
        }
        return out
    }

    private fun flattenOutput(v: Any?): FloatArray = when (v) {
        is FloatArray -> v
        is Array<*> -> {
            val flat = ArrayList<Float>()
            flattenInto(v, flat)
            FloatArray(flat.size) { flat[it] }
        }
        else -> error("Unsupported ONNX output type: ${v?.let { it::class }}")
    }

    private fun flattenInto(a: Array<*>, out: ArrayList<Float>) {
        for (e in a) {
            when (e) {
                is FloatArray -> for (f in e) out.add(f)
                is Array<*> -> flattenInto(e, out)
                is Float -> out.add(e)
                else -> error("Unsupported nested ONNX output element: ${e?.let { it::class }}")
            }
        }
    }
}
