import AppKit
import ApplicationServices

/// Reading text fields of other apps through the Accessibility API.
enum AXText {
    static func appElement(_ pid: pid_t) -> AXUIElement {
        let app = AXUIElementCreateApplication(pid)
        // Electron apps (Slack, VS Code, Orca) only build their accessibility tree when asked.
        AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
        return app
    }

    static func focusedElement(_ pid: pid_t) -> AXUIElement? {
        var v: CFTypeRef?
        guard AXUIElementCopyAttributeValue(appElement(pid), kAXFocusedUIElementAttribute as CFString, &v) == .success,
              let v, CFGetTypeID(v) == AXUIElementGetTypeID() else { return nil }
        return (v as! AXUIElement)
    }

    static func value(_ el: AXUIElement) -> String? {
        var v: CFTypeRef?
        guard AXUIElementCopyAttributeValue(el, kAXValueAttribute as CFString, &v) == .success else { return nil }
        return v as? String
    }

    static func selectionStart(_ el: AXUIElement) -> Int? {
        var v: CFTypeRef?
        guard AXUIElementCopyAttributeValue(el, kAXSelectedTextRangeAttribute as CFString, &v) == .success,
              let v, CFGetTypeID(v) == AXValueGetTypeID() else { return nil }
        var range = CFRange()
        guard AXValueGetValue(v as! AXValue, .cfRange, &range) else { return nil }
        return range.location
    }

    static func windowTitle(_ pid: pid_t) -> String? {
        var w: CFTypeRef?
        guard AXUIElementCopyAttributeValue(appElement(pid), kAXFocusedWindowAttribute as CFString, &w) == .success,
              let w, CFGetTypeID(w) == AXUIElementGetTypeID() else { return nil }
        var t: CFTypeRef?
        guard AXUIElementCopyAttributeValue(w as! AXUIElement, kAXTitleAttribute as CFString, &t) == .success else { return nil }
        return t as? String
    }

    /// Window title plus up to `limit` characters before the cursor, for the post-editor.
    static func context(pid: pid_t, limit: Int = 300) -> String {
        var parts: [String] = []
        if let title = windowTitle(pid), !title.isEmpty { parts.append("窗口：\(title)") }
        if let el = focusedElement(pid), let text = value(el), !text.isEmpty {
            let utf16 = Array(text.utf16)
            let cursor = min(selectionStart(el) ?? utf16.count, utf16.count)
            let start = max(0, cursor - limit)
            if let before = String(utf16CodeUnits: Array(utf16[start..<cursor]), count: cursor - start) as String?,
               !before.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                parts.append("光标前：\(before)")
            }
        }
        return parts.joined(separator: "\n")
    }
}

/// Word-level diff between what Kaikou inserted and what the user turned it into.
enum TokenDiff {
    /// Latin words (with . _ - inside), single CJK characters, or single other characters.
    static func tokens(_ s: String) -> [String] {
        var out: [String] = []
        var word = ""
        func flush() { if !word.isEmpty { out.append(word); word = "" } }
        for ch in s {
            if ch.isASCII && (ch.isLetter || ch.isNumber || ((ch == "." || ch == "_" || ch == "-") && !word.isEmpty)) {
                word.append(ch)
            } else {
                flush()
                out.append(String(ch))
            }
        }
        flush()
        return out
    }

    private static func isContent(_ t: String) -> Bool {
        t.unicodeScalars.contains { CharacterSet.alphanumerics.contains($0) }
    }

    /// Pairs (inserted, corrected) for short replacements; pure additions, deletions and
    /// punctuation-only changes are ignored.
    static func corrections(inserted: String, edited: String, maxTokens: Int = 6) -> [(String, String)] {
        let a = tokens(inserted), b = tokens(edited)
        guard !a.isEmpty, !b.isEmpty, a.count <= 600, b.count <= 800 else { return [] }
        // LCS table.
        var dp = [[Int]](repeating: [Int](repeating: 0, count: b.count + 1), count: a.count + 1)
        for i in stride(from: a.count - 1, through: 0, by: -1) {
            for j in stride(from: b.count - 1, through: 0, by: -1) {
                dp[i][j] = a[i] == b[j] ? dp[i + 1][j + 1] + 1 : max(dp[i + 1][j], dp[i][j + 1])
            }
        }
        // A rewrite, not a fix: learn nothing.
        let contentCount = a.filter(isContent).count
        guard contentCount > 0, Double(dp[0][0]) >= 0.6 * Double(a.count) else { return [] }
        var pairs: [(String, String)] = []
        var i = 0, j = 0
        var del: [String] = [], ins: [String] = []
        var prevEqual: String?
        func isCJK(_ t: String) -> Bool { t.unicodeScalars.contains { (0x3400...0x9FFF).contains($0.value) } }
        func close(next: String?) {
            defer { del.removeAll(); ins.removeAll() }
            let d = del.filter(isContent), n = ins.filter(isContent)
            guard !d.isEmpty, !n.isEmpty, d.count <= maxTokens, n.count <= maxTokens else { return }
            var left = "", right = ""
            // A one-character Chinese fix ("身" -> "审") would rewrite every such character;
            // learn it with its neighbours instead ("自身修" -> "自审修").
            if d.count == 1, isCJK(d[0]) {
                if let p = prevEqual, isCJK(p) { left = p }
                if let q = next, isCJK(q) { right = q }
                guard !left.isEmpty || !right.isEmpty else { return }
            }
            let trim = CharacterSet.whitespacesAndNewlines.union(.punctuationCharacters)
            let from = left + del.joined().trimmingCharacters(in: trim) + right
            let to = left + ins.joined().trimmingCharacters(in: trim) + right
            guard !from.isEmpty, !to.isEmpty, from != to, !to.contains("\n") else { return }
            pairs.append((from, to))
        }
        while i < a.count || j < b.count {
            if i < a.count, j < b.count, a[i] == b[j], a[i].allSatisfy(\.isWhitespace), !(del.isEmpty && ins.isEmpty),
               i + 1 < a.count, j + 1 < b.count, a[i + 1] != b[j + 1] {
                // A space left alone between two changed words does not split the fix
                // ("Put Ted" to "更新tech" is one fix, not "Put" deleted and "Ted" replaced).
                del.append(a[i]); ins.append(b[j])
                i += 1; j += 1
            } else if i < a.count, j < b.count, a[i] == b[j] {
                close(next: a[i])
                prevEqual = a[i]
                i += 1; j += 1
            } else if j < b.count, i == a.count || dp[i][j + 1] >= dp[i + 1][j] {
                ins.append(b[j]); j += 1
            } else {
                del.append(a[i]); i += 1
            }
        }
        close(next: nil)
        return pairs
    }
}

/// Watches the text Kaikou just pasted; if the user fixes words in it, reports the fixes.
final class CorrectionWatcher {
    var onCorrections: (([(String, String)]) -> Void)?
    private var timer: Timer?

    func stop() {
        timer?.invalidate()
        timer = nil
    }

    func watch(inserted: String, pid: pid_t) {
        stop()
        // Give the target app a moment to apply the paste.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in self?.begin(inserted, pid) }
    }

    private func begin(_ inserted: String, _ pid: pid_t) {
        guard let el = AXText.focusedElement(pid), let v0 = AXText.value(el),
              let range = v0.range(of: inserted, options: .backwards) else {
            AppLog.write("learning: pasted text not readable in this app")
            return
        }
        let prefix = String(v0[..<range.lowerBound]), suffix = String(v0[range.upperBound...])
        var last = inserted
        var ticks = 0
        timer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: true) { [weak self] t in
            ticks += 1
            var done = ticks > 80 // about two minutes
            let frontPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            if frontPID != pid {
                done = true // the user moved on to another app
            } else if let current = AXText.focusedElement(pid), CFEqual(current, el), let v = AXText.value(el),
               v.hasPrefix(prefix), v.hasSuffix(suffix), v.count >= prefix.count + suffix.count {
                let start = v.index(v.startIndex, offsetBy: prefix.count)
                let end = v.index(v.endIndex, offsetBy: -suffix.count)
                if start <= end { last = String(v[start..<end]) }
            } else {
                done = true // focus moved, the message was sent, or the user edited elsewhere
            }
            guard done else { return }
            t.invalidate()
            self?.timer = nil
            let pairs = last == inserted ? [] : TokenDiff.corrections(inserted: inserted, edited: last)
            AppLog.write("learning: \(pairs.count) correction(s) found")
            if !pairs.isEmpty { self?.onCorrections?(pairs) }
        }
    }
}
