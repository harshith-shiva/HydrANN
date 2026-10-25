package com.hydrann.ui;

import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.Region;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import javafx.scene.text.TextFlow;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Asset loading and small node factories used across the views. */
final class PixelKit {

    private static final Map<String, Image> IMAGES = new HashMap<>();

    private PixelKit() {}

    static void loadFonts() {
        for (String f : new String[] {"PressStart2P-Regular.ttf", "PixelifySans.ttf"}) {
            try (InputStream in = PixelKit.class.getResourceAsStream("/ui/fonts/" + f)) {
                if (in != null) Font.loadFont(in, 14);
            } catch (IOException ignored) {
                // falls back to the system font
            }
        }
    }

    static String stylesheet() {
        URL u = PixelKit.class.getResource("/ui/theme.css");
        if (u == null) throw new IllegalStateException("Missing resource /ui/theme.css");
        return u.toExternalForm();
    }

    static Image image(String name) {
        return IMAGES.computeIfAbsent(name, n -> {
            URL u = PixelKit.class.getResource("/ui/img/" + n + ".png");
            if (u == null) throw new IllegalStateException("Missing sprite /ui/img/" + n + ".png");
            return new Image(u.toExternalForm());
        });
    }

    /** A sprite at its native size, with smoothing off so every art pixel stays a crisp square. */
    static ImageView sprite(String name) {
        ImageView v = new ImageView(image(name));
        v.setSmooth(false);
        v.setPreserveRatio(true);
        v.setMouseTransparent(true);
        return v;
    }

    static Label label(String text, String styleClass) {
        Label l = new Label(text);
        l.getStyleClass().add(styleClass);
        return l;
    }

    static Region hGrow() {
        Region r = new Region();
        HBox.setHgrow(r, Priority.ALWAYS);
        return r;
    }

    static Region vGrow() {
        Region r = new Region();
        VBox.setVgrow(r, Priority.ALWAYS);
        return r;
    }

    static Text styledText(String s, String styleClass) {
        Text t = new Text(s);
        t.getStyleClass().add(styleClass);
        return t;
    }

    /** Wrapping text where words matching a query token get a highlighter chip. */
    static TextFlow highlightedFlow(String text, List<String> tokens, String textClass) {
        TextFlow flow = new TextFlow();
        flow.getStyleClass().add("flow");
        if (tokens.isEmpty()) {
            flow.getChildren().add(styledText(text, textClass));
            return flow;
        }
        StringBuilder alt = new StringBuilder();
        for (String t : tokens) {
            if (alt.length() > 0) alt.append('|');
            alt.append(Pattern.quote(t));
        }
        Matcher m = Pattern.compile("[\\p{L}\\p{N}]*(?:" + alt + ")[\\p{L}\\p{N}]*",
                Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE).matcher(text);
        int last = 0;
        while (m.find()) {
            if (m.start() > last) flow.getChildren().add(styledText(text.substring(last, m.start()), textClass));
            Label hl = new Label(m.group());
            hl.getStyleClass().add("hl");
            flow.getChildren().add(hl);
            last = m.end();
        }
        if (last < text.length()) flow.getChildren().add(styledText(text.substring(last), textClass));
        return flow;
    }

    static void fixHeight(Region r, double h) {
        r.setMinHeight(h);
        r.setPrefHeight(h);
        r.setMaxHeight(h);
    }

    static void fixWidth(Region r, double w) {
        r.setMinWidth(w);
        r.setPrefWidth(w);
        r.setMaxWidth(w);
    }
}
