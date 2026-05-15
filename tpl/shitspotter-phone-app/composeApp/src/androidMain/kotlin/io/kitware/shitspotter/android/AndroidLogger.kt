package io.kitware.shitspotter.android

import android.util.Log
import io.kitware.shitspotter.core.AppLogger

object AndroidLogger : AppLogger {
    override fun info(tag: String, msg: String) { Log.i("ShitSpotter.$tag", msg) }
    override fun warn(tag: String, msg: String, t: Throwable?) {
        if (t != null) Log.w("ShitSpotter.$tag", msg, t) else Log.w("ShitSpotter.$tag", msg)
    }
    override fun error(tag: String, msg: String, t: Throwable?) {
        if (t != null) Log.e("ShitSpotter.$tag", msg, t) else Log.e("ShitSpotter.$tag", msg)
    }
}
