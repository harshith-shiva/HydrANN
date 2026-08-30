package com.hydrann.ingestion;

public class ExtractedPage {

    private final int pdfPageNumber;
    private final Integer printedPageNumber;
    private final String text;

    public ExtractedPage(
            int pdfPageNumber,
            Integer printedPageNumber,
            String text) {

        this.pdfPageNumber = pdfPageNumber;
        this.printedPageNumber = printedPageNumber;
        this.text = text;
    }

    public int getPdfPageNumber() {
        return pdfPageNumber;
    }

    public Integer getPrintedPageNumber() {
        return printedPageNumber;
    }

    public String getText() {
        return text;
    }
}