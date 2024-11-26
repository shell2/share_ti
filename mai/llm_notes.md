# Notes

## LLM Essential concepts

https://learn.microsoft.com/en-us/dotnet/ai/

<h2 class="margin-none font-size-h6">Essential concepts</h2>
  <h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
    <span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-concept" aria-hidden="true"></span>
    Concept
  </h3>
  <ul class="margin-none has-line-height-reset">
    <li class="is-unstyled padding-block-xxs">
      <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/how-genai-and-llms-work" data-linktype="relative-path">How generative AI and LLMs work</a>
    </li>
    <li class="is-unstyled padding-block-xxs">
      <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/understanding-tokens" data-linktype="relative-path">Understand tokens</a>
    </li>
    <li class="is-unstyled padding-block-xxs">
      <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/embeddings" data-linktype="relative-path">Preserve semantic meaning with embeddings</a>
    </li>
    <li class="is-unstyled padding-block-xxs">
      <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/vector-databases" data-linktype="relative-path">Semantic search with vector databases</a>
    </li>
    <li class="is-unstyled padding-block-xxs">
      <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/prompt-engineering-dotnet" data-linktype="relative-path">Prompt engineering</a>
    </li>
  </ul>

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/how-genai-and-llms-work" data-linktype="relative-path">How generative AI and LLMs work</a>

https://learn.microsoft.com/en-us/dotnet/ai/conceptual/how-genai-and-llms-work

When training an LLM, the training text is first broken down into tokens. Each token identifies a unique text value. A token can be a distinct word, a partial word, or a combination of words and punctuation. Each token is assigned an ID, which enables the text to be represented as a sequence of token IDs.

> After the text has been broken down into tokens, a contextual vector, known as an embedding, is assigned to each token. These embedding vectors are multi-valued numeric data where each element of a token's vector represents a semantic attribute of the token. The elements of a token's vector are determined based on how commonly tokens are used together or in similar contexts.

- summarized by Copilot

Sure! Here are the main points from the page:

### **How Generative AI and LLMs Work**
- **Generative AI**: Creates original content (text, images, audio, code) based on user inputs.
- **Large Language Models (LLMs)**: Used for natural language processing (NLP) to generate text, classify input, and determine sentiment.

### **Common Uses of Generative AI**
- **Natural Language Generation**: Produces text outputs like summaries, descriptions, and ideas.
- **Image Generation**: Creates images from text descriptions, useful for avatars, logos, and artistic inspiration.
- **Audio Generation**: Synthesizes voices, creates music, and modifies audio files.
- **Code Generation**: Generates code snippets in various languages based on user input.

### **How LLMs Work**
- **Tokenization**: Text is broken down into tokens, each assigned an ID.
- **Embeddings**: Tokens are represented as vectors, predicting the next token in a sequence.
- **Training**: Adjusts weights to reduce loss and improve model accuracy.

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/understanding-tokens" data-linktype="relative-path">Understand tokens</a>

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/embeddings" data-linktype="relative-path">Preserve semantic meaning with embeddings</a>

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/vector-databases" data-linktype="relative-path">Semantic search with vector databases</a>

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/prompt-engineering-dotnet" data-linktype="relative-path">Prompt engineering</a>
