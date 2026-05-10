package io.kitware.shitspotter.core

interface AppLogger {
    fun info(tag: String, msg: String)
    fun warn(tag: String, msg: String, t: Throwable? = null)
    fun error(tag: String, msg: String, t: Throwable? = null)
}

object PrintlnLogger : AppLogger {
    override fun info(tag: String, msg: String) = println("[I/$tag] $msg")
    override fun warn(tag: String, msg: String, t: Throwable?) {
        println("[W/$tag] $msg${t?.let { " — ${it.message}" } ?: ""}")
    }
    override fun error(tag: String, msg: String, t: Throwable?) {
        println("[E/$tag] $msg${t?.let { " — ${it.message}" } ?: ""}")
    }
}
