package com.hydrann;

import com.hydrann.ingestion.ExtractedPdf;
import com.hydrann.ingestion.ExtractedPage;
import com.hydrann.ingestion.PdfExtractor;

import java.nio.file.Path;

public class Main {

    public static void main(String[] args) throws Exception {

        // Path to the PDF
        Path pdfPath = Path.of("data/raw-docs/Lab Exercise-2_OR DB.pdf");
        
        // Create extractor
        PdfExtractor extractor = new PdfExtractor();

        // Extract the PDF
        ExtractedPdf pdf = extractor.extract(pdfPath);

        // Print metadata
        System.out.println("===== PDF INFORMATION =====");
        System.out.println("Document ID : " + pdf.getDocumentId());
        System.out.println("File name   : " + pdf.getFileName());
        System.out.println("File path   : " + pdf.getFilePath());
        System.out.println("Title       : " + pdf.getTitle());
        System.out.println("Author      : " + pdf.getAuthor());
        System.out.println("Subject     : " + pdf.getSubject());
        System.out.println("Page count  : " + pdf.getPageCount());

        // Print extracted text page by page
        System.out.println("\n===== EXTRACTED TEXT =====");

        for (ExtractedPage page : pdf.getPages()) {

            System.out.println(
                    "\n----- PDF PAGE "
                    + page.getPdfPageNumber()
                    + " -----"
            );

            System.out.println(page.getText());
        }
    }
}