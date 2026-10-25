package com.hydrann.ui;

import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import java.util.List;
import java.util.Locale;

/** Which search algorithm is live right now, why, and how it is performing. */
final class IndexLabView extends VBox {

    private final Label headType = PixelKit.label("-", "lab-head");
    private final Label headName = PixelKit.label("", "card-meta");
    private final Label phase = PixelKit.label("STABLE", "phase-badge");
    private final Label explain = PixelKit.label("", "lab-explain");

    private final Label recall = PixelKit.label("-", "tile-value");
    private final Label latency = PixelKit.label("-", "tile-value");
    private final Label speed = PixelKit.label("-", "tile-value");
    private final Label memory = PixelKit.label("-", "tile-value");
    private final Label recallSub = PixelKit.label("", "muted-small");
    private final Label latencySub = PixelKit.label("", "muted-small");
    private final Label speedSub = PixelKit.label("", "muted-small");
    private final Label memorySub = PixelKit.label("", "muted-small");

    private final Label corpus = PixelKit.label("", "muted");
    private final Label activity = PixelKit.label("", "muted");
    private final Label decision = PixelKit.label("", "lab-explain");
    private final VBox board = new VBox(4);
    private final VBox events = new VBox(4);

    IndexLabView() {
        getStyleClass().add("page");
        setSpacing(16);

        HBox title = new HBox(16, PixelKit.sprite("mascot_44"), PixelKit.label("Index Lab", "page-title"));
        title.setAlignment(Pos.CENTER_LEFT);

        // active head
        explain.setWrapText(true);
        phase.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
        HBox headRow = new HBox(14, headType, PixelKit.hGrow(), phase);
        headRow.setAlignment(Pos.CENTER_LEFT);
        VBox active = new VBox(8, PixelKit.label("Running now", "panel-heading"), headRow, headName, explain);
        active.getStyleClass().add("pixel-panel");

        // metric tiles
        FlowPane tiles = new FlowPane(12, 12,
                tile("Recall", recall, recallSub), tile("Latency", latency, latencySub),
                tile("Speed-up", speed, speedSub), tile("Head memory", memory, memorySub));

        decision.setWrapText(true);
        VBox decisionBox = new VBox(8, PixelKit.label("Last decision", "panel-heading"), decision, corpus, activity);
        decisionBox.getStyleClass().add("pixel-panel");

        VBox boardBox = new VBox(8, PixelKit.label("Last competition", "panel-heading"), board);
        boardBox.getStyleClass().add("pixel-panel");
        VBox eventBox = new VBox(8, PixelKit.label("Event log", "panel-heading"), events);
        eventBox.getStyleClass().add("pixel-panel");

        VBox content = new VBox(14, active, tiles, decisionBox, boardBox, eventBox);
        content.setPadding(new Insets(2, 10, 8, 2));
        ScrollPane sp = new ScrollPane(content);
        sp.setFitToWidth(true);
        sp.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
        sp.getStyleClass().add("pixel-scroll");
        VBox.setVgrow(sp, Priority.ALWAYS);

        getChildren().addAll(title, sp);
    }

    private static VBox tile(String key, Label value, Label sub) {
        VBox t = new VBox(4, PixelKit.label(key, "stat-key"), value, sub);
        t.getStyleClass().add("tile");
        PixelKit.fixWidth(t, 200);
        return t;
    }

    void update(SearchBackend.HydraInfo h) {
        headType.setText(h.headType().toUpperCase(Locale.ROOT));
        headName.setText(h.headName());
        phase.setText(h.phase() + (h.phaseNote().isBlank() ? "" : "  \u00B7  " + h.phaseNote()));
        phase.getStyleClass().setAll("phase-badge", "phase-" + h.phase().toLowerCase(Locale.ROOT));
        explain.setText(describe(h.headType()));

        if (h.samples() == 0) {
            recall.setText("-");
            latency.setText("-");
            speed.setText("-");
            recallSub.setText("run a few searches");
            latencySub.setText("");
            speedSub.setText("");
        } else {
            recall.setText(String.format(Locale.ROOT, "%.2f", h.recall()));
            latency.setText(fmtUs(h.avgLatencyUs()));
            speed.setText(String.format(Locale.ROOT, "%.1fx", h.speedup()));
            recallSub.setText("over " + h.samples() + " checked queries");
            latencySub.setText("p95 " + fmtUs(h.p95LatencyUs()));
            speedSub.setText("vs brute force " + fmtUs(h.bruteAvgUs()));
        }
        memory.setText(formatBytes(h.memoryBytes()));
        memorySub.setText("on this head " + formatDuration(System.currentTimeMillis() - h.activeSinceMs()));

        decision.setText(h.lastDecisionAtMs() == 0 ? "No competition has run yet."
                : formatDuration(System.currentTimeMillis() - h.lastDecisionAtMs()) + " ago: " + h.lastDecision());
        corpus.setText(String.format(Locale.ROOT, "Corpus: %d active of %d stored passages", h.activeVectors(), h.storedVectors()));
        activity.setText(count(h.promotions(), "switch", "switches") + "  \u00B7  " + count(h.competitions(), "competition", "competitions")
                + "  \u00B7  " + count(h.tunings(), "tuning", "tunings"));

        fill(board, h.scoreboard(), "mono-line", "Waiting for the first competition...");
        fill(events, h.events(), "mono-line", "Nothing yet.");
    }

    private static String count(int n, String one, String many) {
        return n + " " + (n == 1 ? one : many);
    }

    private static void fill(VBox box, List<String> lines, String style, String emptyText) {
        box.getChildren().clear();
        if (lines.isEmpty()) {
            box.getChildren().add(PixelKit.label(emptyText, "muted"));
            return;
        }
        for (String l : lines) {
            Label lab = PixelKit.label(l, style);
            if (l.startsWith(">")) lab.getStyleClass().add("mono-winner");
            box.getChildren().add(lab);
        }
    }

    private static String describe(String type) {
        return switch (type.toLowerCase(Locale.ROOT)) {
            case "brute-force" -> "Compares your question with every passage. Always exact; best while the library is small.";
            case "lsh" -> "Hashes passages into buckets with random hyperplanes and only checks the matching buckets.";
            case "hnsw" -> "Walks a layered graph of nearest neighbours. Fast and accurate on large libraries.";
            case "ivf" -> "Groups passages into clusters and only searches the clusters closest to your question.";
            default -> "";
        };
    }

    static String fmtUs(double us) {
        return us >= 1000 ? String.format(Locale.ROOT, "%.1f ms", us / 1000.0) : String.format(Locale.ROOT, "%.0f us", us);
    }

    static String formatBytes(long b) {
        if (b < 1024) return b + " B";
        if (b < 1024L * 1024) return (b / 1024) + " KB";
        return String.format(Locale.ROOT, "%.1f MB", b / (1024.0 * 1024));
    }

    static String formatDuration(long ms) {
        if (ms < 1_000) return ms + " ms";
        if (ms < 60_000) return String.format(Locale.ROOT, "%.0f s", ms / 1000.0);
        long s = ms / 1000;
        if (s < 3600) return (s / 60) + "m " + String.format(Locale.ROOT, "%02d", s % 60) + "s";
        return (s / 3600) + "h " + ((s % 3600) / 60) + "m";
    }
}
