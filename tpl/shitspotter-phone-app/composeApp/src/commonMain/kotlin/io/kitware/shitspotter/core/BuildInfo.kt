package io.kitware.shitspotter.core

expect object BuildInfo {
    val deviceModel: String
    val osVersion: String
    val appCommit: String
    val buildDate: String
}
