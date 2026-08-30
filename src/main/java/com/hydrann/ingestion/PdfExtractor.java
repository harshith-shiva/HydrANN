package com.hydrann.ingestion;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.PDDocumentInformation;
import org.apache.pdfbox.text.PDFTextStripper;

public class PdfExtractor {

    public ExtractedPdf extract(Path pdfPath) throws IOException {

        // 1. Validate path
        if (pdfPath == null) {
            throw new IllegalArgumentException(
                    "PDF path cannot be null");
        }

        if (!Files.exists(pdfPath)) {
            throw new IOException(
                    "PDF file does not exist: " + pdfPath);
        }

        if (!Files.isRegularFile(pdfPath)) {
            throw new IOException(
                    "Path is not a file: " + pdfPath);
        }

        
        String documentId = UUID.randomUUID().toString();

        
        try (PDDocument document = Loader.loadPDF(pdfPath.toFile())) {

            int pageCount = document.getNumberOfPages();

            
            PDDocumentInformation info =
                    document.getDocumentInformation();

            String title = info.getTitle();
            String author = info.getAuthor();
            String subject = info.getSubject();

            // 5. Extract each page
            List<ExtractedPage> pages = new ArrayList<>();

            PDFTextStripper stripper = new PDFTextStripper();

            for (int i = 0; i < pageCount; i++) {

                
                int pdfPageNumber = i + 1;

                stripper.setStartPage(pdfPageNumber);
                stripper.setEndPage(pdfPageNumber);

                String text = stripper.getText(document);

                ExtractedPage page = new ExtractedPage(
                        pdfPageNumber,
                        null,
                        text
                );
                // System.out.println(page.getText());

                pages.add(page);

            }


            return new ExtractedPdf(
                    documentId,
                    pdfPath.getFileName().toString(),
                    pdfPath.toAbsolutePath(),
                    title,
                    author,
                    subject,
                    pageCount,
                    pages
            );
        }
    }
}