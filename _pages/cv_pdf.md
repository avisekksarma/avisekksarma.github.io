---
layout: page
permalink: /cv/pdf/
title: Resume
nav: false
description: PDF résumé — view in the browser or download.
---

{% include cv/styles.liquid %}

<div class="pdf-page">
  <p class="pdf-toolbar">
    <a class="pdf-btn" href="{{ '/assets/pdf/Abhishek_Sharma_Resume.pdf' | relative_url }}" download="Abhishek_Sharma_Resume.pdf">Download PDF</a>
    <a class="pdf-btn" href="{{ '/assets/pdf/Abhishek_Sharma_Resume.pdf' | relative_url }}" target="_blank" rel="noopener">Open in new tab</a>
    <a class="pdf-btn pdf-btn-quiet" href="{{ '/cv/' | relative_url }}">HTML version</a>
  </p>

  <iframe
    class="pdf-frame"
    src="{{ '/assets/pdf/Abhishek_Sharma_Resume.pdf' | relative_url }}#view=FitH"
    title="Abhishek Sharma résumé"
  ></iframe>
</div>
