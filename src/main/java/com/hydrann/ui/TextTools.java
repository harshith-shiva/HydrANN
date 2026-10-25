package com.hydrann.ui;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** Small string helpers shared by the backends and the views. */
final class TextTools {

    private static final Set<String> STOP = Set.of(
            "the", "and", "for", "are", "was", "were", "with", "that", "this", "from", "what", "how",
            "does", "can", "why", "who", "which", "when", "where", "into", "about", "have", "has",
            "had", "not", "but", "its", "you", "your", "our", "their", "them", "they", "there",
            "than", "then", "also", "been", "will", "would", "could", "should");

    private TextTools() {}

    static String fileName(String path) {
        if (path == null) return "";
        int i = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
        return i < 0 ? path : path.substring(i + 1);
    }

    /** "ncert_science.pdf" becomes "Ncert Science". */
    static String prettyName(String fileName) {
        String base = fileName;
        int dot = base.lastIndexOf('.');
        if (dot > 0) base = base.substring(0, dot);
        return titleCase(base.replaceAll("[-_]+", " ").replaceAll("\\s+", " ").trim());
    }

    static String titleCase(String s) {
        StringBuilder sb = new StringBuilder();
        for (String w : s.split(" ")) {
            if (w.isEmpty()) continue;
            if (sb.length() > 0) sb.append(' ');
            sb.append(Character.toUpperCase(w.charAt(0)));
            if (w.length() > 1) sb.append(w.substring(1).toLowerCase(Locale.ROOT));
        }
        return sb.toString();
    }

    static String collapse(String s) {
        return s == null ? "" : s.replaceAll("\\s+", " ").trim();
    }

    /** Lower-case content words of a query (3+ letters, no stop words). */
    static List<String> queryTokens(String query) {
        Set<String> out = new LinkedHashSet<>();
        for (String w : query.toLowerCase(Locale.ROOT).split("[^\\p{L}\\p{N}]+")) {
            if (w.length() >= 3 && !STOP.contains(w)) out.add(w);
        }
        return new ArrayList<>(out);
    }

    /** A window of about {@code max} characters, centred near the first query word. */
    static String snippetAround(String text, List<String> tokens, int max) {
        String flat = collapse(text);
        if (flat.length() <= max) return flat;

        String lower = flat.toLowerCase(Locale.ROOT);
        int hit = -1;
        for (String t : tokens) {
            int i = lower.indexOf(t);
            if (i >= 0 && (hit < 0 || i < hit)) hit = i;
        }
        int start = hit < 0 ? 0 : Math.max(0, hit - max / 4);
        if (start > 0) {
            int sp = flat.indexOf(' ', start);
            if (sp > 0 && sp < start + 20) start = sp + 1;
        }
        int end = Math.min(flat.length(), start + max);
        if (end < flat.length()) {
            int sp = flat.lastIndexOf(' ', end);
            if (sp > start + max / 2) end = sp;
        }
        return (start > 0 ? "... " : "") + flat.substring(start, end).trim() + (end < flat.length() ? " ..." : "");
    }

    /**
     * Picks a heading-like line from the start of a chunk ("ANIMAL KINGDOM 5353" becomes
     * "Animal Kingdom"); falls back to the document's prettified file name.
     */
    static String deriveTitle(String text, String fileName) {
        if (text != null) {
            String[] lines = text.split("\\R");
            for (int i = 0; i < Math.min(4, lines.length); i++) {
                String c = lines[i].replaceFirst("^[\\d\\W_]+", "").replaceFirst("[\\s\\d]+$", "").trim();
                if (c.length() < 6 || c.length() > 60) continue;
                if (!Character.isUpperCase(c.charAt(0))) continue;
                char last = c.charAt(c.length() - 1);
                if (last == '.' || last == ',' || last == ';' || last == ':') continue;
                long letters = c.chars().filter(Character::isLetter).count();
                if (letters < c.length() * 0.7) continue;
                boolean allCaps = c.equals(c.toUpperCase(Locale.ROOT));
                if (!allCaps && c.indexOf(' ') < 0) continue;
                return allCaps ? titleCase(c) : c;
            }
        }
        return prettyName(fileName);
    }
}
