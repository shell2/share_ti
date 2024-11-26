# Notes

LLM Models: Gemini (Google), Llama3 (Meta), Claude (Anthropic), GPT-4 (Openai & Microsoft), Mixtral-8x22B

Sure! Here are some basic AI concepts explained:

### **1. Model**
An AI model is a mathematical representation of a real-world process. It is trained on data to recognize patterns and make predictions or decisions. Models can be used for various tasks like image recognition, language translation, and more.

### **2. Model Tuning**
Model tuning involves adjusting the parameters of an AI model to improve its performance. This can include:
- **Hyperparameter Tuning**: Adjusting settings like learning rate, batch size, etc.
- **Fine-Tuning**: Retraining a pre-trained model on a new, specific dataset to adapt it to a particular task.

### **3. Retrieval-Augmented Generation (RAG)**
RAG combines retrieval-based and generative approaches to enhance the accuracy of AI models by incorporating external knowledge. It involves:
- **Retrieval Component**: Fetches relevant information from external sources.
- **Generation Component**: Uses the retrieved information to generate a more accurate and contextually relevant response¹(https://learnbybuilding.ai/tutorials/rag-from-scratch)²(https://towardsai.net/p/machine-learning/optimizing-ai-models-with-fine-tuning-and-rag-which-approach-wins).

### **4. Agent**
In AI, an agent is an entity that perceives its environment through sensors and acts upon it using actuators. Agents can be:
- **Reactive**: Responds to changes in the environment.
- **Proactive**: Takes actions to achieve specific goals.

### **5. Fine-Tuning vs. RAG**
- **Fine-Tuning**: Adapts a pre-trained model to a specific task by training it on a specialized dataset²(https://towardsai.net/p/machine-learning/optimizing-ai-models-with-fine-tuning-and-rag-which-approach-wins).
- **RAG**: Enhances model responses by integrating real-time information from external sources¹(https://learnbybuilding.ai/tutorials/rag-from-scratch)²(https://towardsai.net/p/machine-learning/optimizing-ai-models-with-fine-tuning-and-rag-which-approach-wins).

These concepts are fundamental to understanding how AI systems are built and optimized for various applications. Let me know if you need more details on any of these topics!

Source: Conversation with Copilot, 27/11/2024
(1) A beginner's guide to building a Retrieval Augmented Generation (RAG .... https://learnbybuilding.ai/tutorials/rag-from-scratch.
(2) Optimizing AI Models with Fine-Tuning and RAG — Which Approach Wins?. https://towardsai.net/p/machine-learning/optimizing-ai-models-with-fine-tuning-and-rag-which-approach-wins.
(3) Adapting AI Models - Processica. https://www.processica.com/articles/adapting-ai-models-the-strategic-choice-between-fine-tuning-and-rag/.
(4) The Potential of Retrieval Augmented Generation (RAG) and Fine-Tuning .... https://dataloop.ai/blog/the-potential-of-retrieval-augmented-generation-rag-and-fine-tuning/.
(5) RAG vs. Fine-Tuning: How to Choose - Oracle. https://www.oracle.com/artificial-intelligence/generative-ai/retrieval-augmented-generation-rag/rag-fine-tuning/.
(6) undefined. https://github.com/jmorganca/ollama/blob/main/docs/api.md.

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

TODO: add
also refer to: Explained: Tokens and Embeddings in LLMs
https://medium.com/the-research-nest/explained-tokens-and-embeddings-in-llms-69a16ba5db33


## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/embeddings" data-linktype="relative-path">Preserve semantic meaning with embeddings</a>

TODO: add

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/vector-databases" data-linktype="relative-path">Semantic search with vector databases</a>

TODO: add

## <a class="has-external-link-indicator font-size-sm display-block" href="conceptual/prompt-engineering-dotnet" data-linktype="relative-path">Prompt engineering</a>

TODO: add

## LLM model perforance compare

refer to: [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://github.com/microsoft/LongRoPE)

## >>> Microsoft AI Playbook

https://learn.microsoft.com/en-us/ai/playbook/

<section id="landing-head">
	<div class="padding-top-xxs padding-bottom-xs">
		<div class="columns">
			<div class="column is-full">
				<h1 class="font-size-h2">AI playbook</h1>
					<p class="margin-top-xxs has-line-height-reset">The Artificial Intelligence (AI) Playbook provides enterprise software engineers with solutions, capabilities, and code developed to solve real-world AI problems.</p>
			</div>
		</div>
	</div>
</section>

<section id="landing-content" class="padding-block-xs">
	<div class="columns is-masonry is-three-masonry-columns">
			<div class="column is-12 is-4-desktop">
				<div class="box box-shadow-medium margin-none margin-xxs-desktop">
					<h2 class="margin-none font-size-h6">Working with Large Language Models (LLMs)</h2>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-overview" aria-hidden="true"></span>
							Overview
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/working-with-llms/" data-linktype="relative-path">Overview of Large Language Models (LLMs)</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/working-with-llms/experiment-recommend" data-linktype="relative-path">Experimenting with LLMs</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-concept" aria-hidden="true"></span>
							Concept
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/getting-started/use-case-recommend" data-linktype="relative-path">Understanding large language models (LLMs)</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/working-with-llms/evaluation/list-of-eval-metrics" data-linktype="relative-path">Defining and understanding LLM evaluation metrics</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/working-with-llms/evaluation/g-eval-metric-for-summarization" data-linktype="relative-path">Evaluating the performance of LLM summarization prompts</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/mlops-in-openai/security/security-recommend" data-linktype="relative-path">Security guidance for Large Language Models</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-get-started" aria-hidden="true"></span>
							Get started
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/working-with-llms/prompt-engineering" data-linktype="relative-path">Getting started with LLM prompt engineering</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/working-with-llms/fine-tuning" data-linktype="relative-path">Getting started with LLM fine-tuning</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-architecture" aria-hidden="true"></span>
							Architecture
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="solutions/generative-ai/rag-experiment-accelerator" data-linktype="relative-path">RAG experiment accelerator</a>
							</li>
						</ul>
				</div>
			</div>
			<div class="column is-12 is-4-desktop">
				<div class="box box-shadow-medium margin-none margin-xxs-desktop">
					<h2 class="margin-none font-size-h6">Generative AI solutions</h2>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-concept" aria-hidden="true"></span>
							Concept
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/" data-linktype="relative-path">Creating production-ready GenAI applications</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/dev-starters/genai-gateway/" data-linktype="relative-path">Designing and implementing Azure OpenAI Gateway</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/dev-starters/genai-gateway/cost-optimization" data-linktype="relative-path">Azure OpenAI cost optimization</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-architecture" aria-hidden="true"></span>
							Architecture
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="solutions/generative-ai/genai-gateway/" data-linktype="relative-path">GenAI Gateway reference architecture</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/generative-ai/dev-starters/genai-gateway/reference-architectures/apim-based" data-linktype="relative-path">GenAI Gateway reference architecture using APIM</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="solutions/generative-ai/genai-gateway/maximise-ptu-utilization" data-linktype="relative-path">Approaches for maximizing PTU utilization</a>
							</li>
						</ul>
				</div>
			</div>
			<div class="column is-12 is-4-desktop">
				<div class="box box-shadow-medium margin-none margin-xxs-desktop">
					<h2 class="margin-none font-size-h6">AI engineering</h2>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-get-started" aria-hidden="true"></span>
							Get started
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/experimentation/" data-linktype="relative-path">Getting started with AI experimentation</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-concept" aria-hidden="true"></span>
							Concept
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/" data-linktype="relative-path">Understanding production AI building blocks</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/data-curation/" data-linktype="relative-path">Understanding data curation and management for AI projects</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/deployment/" data-linktype="relative-path">AI model deployment</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/ml-lifecycle-management/" data-linktype="relative-path">Using MLOps to manage the AI lifecycle</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/mlops/drift-overview" data-linktype="relative-path">Understanding and adapting to data drift</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-reference" aria-hidden="true"></span>
							Reference
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/model-development/" data-linktype="relative-path">capabilities/model-development</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="technology-guidance/mlops/drift-overview" data-linktype="relative-path">How to detect and adapt to drift</a>
							</li>
						</ul>
				</div>
			</div>
			<div class="column is-12 is-4-desktop">
				<div class="box box-shadow-medium margin-none margin-xxs-desktop">
					<h2 class="margin-none font-size-h6">AI experimentation</h2>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-get-started" aria-hidden="true"></span>
							Get started
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="capabilities/experimentation/" data-linktype="relative-path">Getting started with AI experimentation</a>
							</li>
						</ul>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-concept" aria-hidden="true"></span>
							Concept
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="https://microsoft.github.io/code-with-engineering-playbook/machine-learning/responsible-ai" data-linktype="external">Responsible AI</a>
							</li>
						</ul>
				</div>
			</div>
			<div class="column is-12 is-4-desktop">
				<div class="box box-shadow-medium margin-none margin-xxs-desktop">
					<h2 class="margin-none font-size-h6">Machine Learning Solutions</h2>
						<h3 class="display-flex is-uppercase font-size-sm border-top margin-bottom-xxs margin-top-xs padding-top-xs has-text-subtle">
							<span class="margin-right-xxs align-self-center font-size-h5 docon docon-topic-architecture" aria-hidden="true"></span>
							Architecture
						</h3>
						<ul class="margin-none has-line-height-reset">
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="solutions/data-discovery/" data-linktype="relative-path">Data Discovery solution for unstructured data</a>
							</li>
							<li class="is-unstyled padding-block-xxs">
								<a class="has-external-link-indicator font-size-sm display-block" href="solutions/custom-machine-learning/automating-model-training/" data-linktype="relative-path">Automating and monitoring ML model development</a>
							</li>
						</ul>
				</div>
			</div>
	</div>
</section>
							<!-- </content> -->

