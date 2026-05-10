package io.kitware.shitspotter.core

import kotlin.test.Test
import kotlin.test.assertFails

class TfliteBackendStubTest {

    @Test
    fun rejects_non_tflite_spec() {
        assertFails {
            TfliteBackendStub(ModelSpec.STUB)
        }
    }

    @Test
    fun warmup_throws_not_implemented() {
        val spec = ModelSpec.STUB.copy(format = ModelFormat.TFLITE)
        val b = TfliteBackendStub(spec)
        assertFails { b.warmup() }
    }

    @Test
    fun analyze_throws_not_implemented() {
        val spec = ModelSpec.STUB.copy(format = ModelFormat.TFLITE)
        val b = TfliteBackendStub(spec)
        val frame = object : FrameSource {
            override val width = 1
            override val height = 1
            override val rotationDegrees = 0
            override fun toRgb888() = ByteArray(3)
        }
        assertFails { b.analyze(frame) }
    }
}
