package com.hydrann.annheads;

/** A head with a query-time "effort" knob that can change without a rebuild. */
public interface Tunable {
    int getEffort();
    void setEffort(int effort);
    int minEffort();
    int maxEffort();
    String effortName();
}