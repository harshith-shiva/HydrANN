package com.hydrann.controller;

import java.util.List;

public record HydraStatus(
        Phase phase, String activeHead, String phaseNote,
        int activeVectors, int storedVectors,
        int samples, double recall, double avgLatencyUs, double p95LatencyUs, double bruteAvgUs,
        long memoryBytes, long activeSinceMs,
        int promotions, int competitions, int tunings,
        String lastDecision, long lastDecisionAtMs,
        List<String> scoreboard, List<String> recentEvents) {

    public enum Phase { STABLE, TUNING, COMPETING }

    public double ghostFraction() {
        return storedVectors == 0 ? 0 : 1.0 - (double) activeVectors / storedVectors;
    }

    public double speedup() {
        return avgLatencyUs > 0 && bruteAvgUs > 0 ? bruteAvgUs / avgLatencyUs : 0;
    }

    public String oneLine() {
        String stats = samples == 0
                ? "no samples yet"
                : String.format("recall %.2f | avg %,.0f us | %.1fx vs brute", recall, avgLatencyUs, speedup());
        return String.format("[%s] %s | %,d active (%.0f%% ghosts) | %s | %s",
                phase, activeHead, activeVectors, ghostFraction() * 100, stats, formatBytes(memoryBytes));
    }

    public String format() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n==================== HYDRA STATUS ====================\n");
        sb.append(String.format("Active head  : %s%n", activeHead));
        sb.append(String.format("Phase        : %s%s%n", phase, phaseNote.isBlank() ? "" : " - " + phaseNote));
        sb.append(String.format("Corpus       : %,d active / %,d stored (%.0f%% ghosts)%n",
                activeVectors, storedVectors, ghostFraction() * 100));
        if (samples == 0) {
            sb.append("Rolling      : no samples yet\n");
        } else {
            sb.append(String.format("Rolling      : recall %.2f | avg %,.0f us | p95 %,.0f us | brute %,.0f us | %.1fx (%d samples)%n",
                    recall, avgLatencyUs, p95LatencyUs, bruteAvgUs, speedup(), samples));
        }
        sb.append(String.format("Resources    : head memory %s | on this head for %s%n",
                formatBytes(memoryBytes), formatDuration(System.currentTimeMillis() - activeSinceMs)));
        sb.append(String.format("Counters     : %d promotions, %d competitions, %d tunings%n",
                promotions, competitions, tunings));
        sb.append("Last decision: ")
          .append(lastDecisionAtMs == 0
                  ? "none yet"
                  : formatDuration(System.currentTimeMillis() - lastDecisionAtMs) + " ago - " + lastDecision)
          .append("\n");
        if (!scoreboard.isEmpty()) {
            sb.append("Scoreboard (last competition):\n");
            for (String line : scoreboard) sb.append("  ").append(line).append("\n");
        }
        if (!recentEvents.isEmpty()) {
            sb.append("Recent events:\n");
            for (String e : recentEvents) sb.append("  ").append(e).append("\n");
        }
        sb.append("======================================================\n");
        return sb.toString();
    }

    public static String formatBytes(long b) {
        if (b < 1024) return b + " B";
        if (b < 1024L * 1024) return String.format("%.0f KB", b / 1024.0);
        return String.format("%.1f MB", b / (1024.0 * 1024));
    }

    public static String formatDuration(long ms) {
        if (ms < 1_000) return ms + " ms";
        if (ms < 60_000) return String.format("%.1f s", ms / 1000.0);
        long s = ms / 1000;
        return String.format("%dm %02ds", s / 60, s % 60);
    }
}