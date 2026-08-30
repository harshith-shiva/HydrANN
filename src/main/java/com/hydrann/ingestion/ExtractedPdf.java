package com.hydrann.ingestion;

import java.nio.file.Path;
import java.util.List;

public class ExtractedPdf {

    private final String documentId;
    private final String fileName;
    private final Path filePath;

    private final String title;
    private final String author;
    private final String subject;

    private final int pageCount;

    private final List<ExtractedPage> pages;

    public ExtractedPdf(
            String documentId,
            String fileName,
            Path filePath,
            String title,
            String author,
            String subject,
            int pageCount,
            List<ExtractedPage> pages) {

        this.documentId = documentId;
        this.fileName = fileName;
        this.filePath = filePath;
        this.title = title;
        this.author = author;
        this.subject = subject;
        this.pageCount = pageCount;
        this.pages = pages;
    }

    public String getDocumentId() {
        return documentId;
    }

    public String getFileName() {
        return fileName;
    }

    public Path getFilePath() {
        return filePath;
    }

    public String getTitle() {
        return title;
    }

    public String getAuthor() {
        return author;
    }

    public String getSubject() {
        return subject;
    }

    public int getPageCount() {
        return pageCount;
    }

    public List<ExtractedPage> getPages() {
        return pages;
    }
}