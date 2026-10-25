package com.hydrann.ui;

import javafx.geometry.Side;
import javafx.scene.control.Button;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.ContextMenu;
import javafx.scene.control.MenuItem;

import java.util.List;
import java.util.function.Consumer;

/** A button that opens a small pixel-styled menu of choices. */
final class PixelDropdown extends Button {

    private final String prefix;
    private final ContextMenu menu = new ContextMenu();
    private String value;
    private Consumer<String> onChange = v -> {};

    PixelDropdown(String prefix, List<String> options, String initial) {
        this.prefix = prefix;
        getStyleClass().add("dropdown");
        setContentDisplay(ContentDisplay.RIGHT);
        setGraphic(PixelKit.sprite("icon_chevron"));
        setGraphicTextGap(10);
        for (String o : options) {
            MenuItem item = new MenuItem(o);
            item.setOnAction(e -> {
                setValue(o);
                onChange.accept(o);
            });
            menu.getItems().add(item);
        }
        setOnAction(e -> {
            if (menu.isShowing()) menu.hide();
            else menu.show(this, Side.BOTTOM, 0, 4);
        });
        setValue(initial);
    }

    String value() {
        return value;
    }

    void setValue(String v) {
        value = v;
        setText(prefix + v);
    }

    void setOnChange(Consumer<String> c) {
        this.onChange = c;
    }
}
