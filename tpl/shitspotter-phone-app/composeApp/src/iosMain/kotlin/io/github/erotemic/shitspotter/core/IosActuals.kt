package io.github.erotemic.shitspotter.core

import platform.Foundation.NSDate
import platform.Foundation.timeIntervalSince1970
import platform.UIKit.UIDevice

actual fun nowMonoMs(): Double = NSDate().timeIntervalSince1970 * 1000.0

actual object BuildInfo {
    actual val deviceModel: String = UIDevice.currentDevice.model
    actual val osVersion: String = "iOS ${UIDevice.currentDevice.systemVersion}"
    actual val appCommit: String = "ios-unset"
    actual val buildDate: String = "ios-unset"
}
