import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from threading import Lock
from .gpu_detector import get_optimal_device

SUBTITLE_REFINEMENT_PROMPTS = {
    'zh': {
        'system': """你是一个专业的ASR（自动语音识别）字幕校对专家，专注于修正语音识别错误。
**核心任务**：
1. **同音字/近音字纠错**：识别并修正同音或近音导致的错误
2. **词语边界修正**：正确识别词语边界，修正分词错误
3. **语法修正**：修正明显的语法错误
4. **口语转书面**：适度优化口语表达
5. **标点规范**：添加或修正标点符号，提升可读性

**重要原则**：
✓ 只修正明确的ASR错误，不过度改写
✓ 保持原意和说话风格
✓ 利用上下文理解语义
✓ 不添加原文不存在的内容
✓ 不确定时保持原样

**输出要求**：
- 直接输出修正后的字幕
- 不要包含任何解释、分析或思考过程
- 不要使用<think>标签或其他标记
- 不要添加"修正后："等前缀
- 只输出最终结果""",
        'user_with_context': """【上下文对话】\n{context}\n\n【当前字幕】\n{text}\n\n根据上下文，修正上述字幕的ASR错误。直接输出修正结果：""",
        'user_no_context': """【需要校对的字幕】\n{text}\n\n修正上述字幕的ASR错误。直接输出修正结果："""
    },
    'en': {
        'system': """You are a professional ASR (Automatic Speech Recognition) subtitle proofreader specializing in correcting speech recognition errors.
**Core Tasks**:
1. **Homophone Correction**: Identify and fix errors caused by homophones
2. **Word Boundary Correction**: Fix word segmentation errors
3. **Grammar Correction**: Fix obvious grammatical errors
4. **Colloquial to Formal**: Moderate optimization
5. **Punctuation**: Add or correct punctuation for clarity

**Important Principles**:
✓ Only fix clear ASR errors, don't over-edit
✓ Maintain original meaning and speaking style
✓ Use context to understand semantics
✓ Don't add content not in original speech
✓ When uncertain, keep original

**Output Requirements**:
- Output the corrected subtitle directly
- No explanations, analysis, or thinking process
- No <think> tags or other markers
- No prefixes like "Corrected:" or "Result:"
- Only output the final result""",
        'user_with_context': """[Context Dialogue]\n{context}\n\n[Current Subtitle]\n{text}\n\nBased on context, correct ASR errors in the subtitle. Output result directly:""",
        'user_no_context': """[Subtitle to Proofread]\n{text}\n\nCorrect ASR errors in the subtitle. Output result directly:"""
    }
}

TRANSLATION_PROMPTS = {
    'zh_to_en': {
        'system': """You are a professional subtitle translator. Translate Chinese subtitles to English directly and concisely.
Rules:
- Output ONLY the English translation
- No explanation, no thinking process
- Keep it natural and concise
- Preserve the tone and emotion""",
        'user': """Translate to English:\n{text}"""
    },
    'en_to_zh': {
        'system': """你是专业字幕翻译助手。直接输出简洁的中文翻译。
规则：
- 只输出中文翻译
- 不要解释、不要思考过程
- 保持自然流畅
- 保留语气和情感""",
        'user': """翻译成中文：\n{text}"""
    }
}

SUPPORTED_TRANSLATION_PAIRS = [
    {"source": "en", "target": "zh", "model": "Helsinki-NLP/opus-mt-en-zh", "name": "English to Chinese"},
    {"source": "en", "target": "fr", "model": "Helsinki-NLP/opus-mt-en-fr", "name": "English to French"},
    {"source": "en", "target": "es", "model": "Helsinki-NLP/opus-mt-en-es", "name": "English to Spanish"},
    {"source": "en", "target": "de", "model": "Helsinki-NLP/opus-mt-en-de", "name": "English to German"},
    {"source": "zh", "target": "en", "model": "Helsinki-NLP/opus-mt-zh-en", "name": "Chinese to English"},
]

class SubtitleTranslator:
    def __init__(self, device=None):
        self.device = device if device else get_optimal_device()[0]
        self.translation_pipelines = {}
        self.qwen_model = None
        self.qwen_tokenizer = None
        self.qwen_lock = Lock()
        self.current_qwen_model_id = None

    def get_supported_pairs(self):
        return SUPPORTED_TRANSLATION_PAIRS

    def get_translation_pipeline(self, model_name):
        if model_name not in self.translation_pipelines:
            try:
                device_idx = 0 if self.device == 'cuda' else -1
                self.translation_pipelines[model_name] = pipeline(
                    "translation", 
                    model=model_name,
                    device=device_idx,
                    max_length=512
                )
            except Exception as e:
                print(f"Failed to load translation model {model_name}: {e}")
                return None
        return self.translation_pipelines[model_name]

    def get_qwen_model(self, model_id="Qwen/Qwen3-4B"):
        with self.qwen_lock:
            if self.qwen_model is not None and self.qwen_tokenizer is not None and self.current_qwen_model_id == model_id:
                return self.qwen_model, self.qwen_tokenizer
            
            try:
                self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
                
                load_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
                if self.device != 'cpu':
                    load_kwargs["torch_dtype"] = torch.float16
                else:
                    load_kwargs["torch_dtype"] = torch.float32
                
                self.qwen_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                if hasattr(self.qwen_model, 'to'):
                    self.qwen_model.to(self.device)
                self.qwen_model.eval()
                
                self.current_qwen_model_id = model_id
                return self.qwen_model, self.qwen_tokenizer
            except Exception as e:
                print(f"Failed to load Qwen model {model_id}: {e}")
                return None, None

    def translate_text(self, text, source_lang, target_lang, model_name=None):
        # 1. Try generic Helsinki models if no specific model
        if not model_name:
            for pair in SUPPORTED_TRANSLATION_PAIRS:
                if pair['source'] == source_lang and pair['target'] == target_lang:
                    model_name = pair['model']
                    break
        
        if not model_name:
             raise ValueError(f"No translation model found for {source_lang} -> {target_lang}")

        pipeline = self.get_translation_pipeline(model_name)
        if pipeline:
            res = pipeline(text)
            return res[0]['translation_text']
        else:
            raise RuntimeError("Translation model failed to load")

    def translate_with_qwen(self, text, source_lang='zh', target_lang='en', context=None, model_name="Qwen/Qwen3-4B"):
        model, tokenizer = self.get_qwen_model(model_name)
        if not model or not tokenizer:
            return text

        prompt_key = None
        if source_lang == 'zh' and target_lang == 'en':
            prompt_key = 'zh_to_en'
        elif source_lang == 'en' and target_lang == 'zh':
            prompt_key = 'en_to_zh'
        
        if not prompt_key:
            return text
            
        prompts = TRANSLATION_PROMPTS[prompt_key]
        messages = [
            {"role": "system", "content": prompts['system']},
            {"role": "user", "content": prompts['user'].format(text=text)}
        ]
        
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt")
        if self.device != 'cpu':
            model_inputs = model_inputs.to(self.device)
            
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=256,
                temperature=0.3, # slightly higher for translation
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    def refine_subtitle(self, text, context=None, language='zh', enable_thinking=False, model_name="Qwen/Qwen3-4B"):
        model, tokenizer = self.get_qwen_model(model_name)
        if not model or not tokenizer:
            return text
            
        lang_key = 'zh' if language in ['zh', 'zh-CN', 'zh-TW'] else 'en'
        prompts = SUBTITLE_REFINEMENT_PROMPTS.get(lang_key, SUBTITLE_REFINEMENT_PROMPTS['en'])
        
        system_prompt = prompts['system']
        if enable_thinking:
            thinking_instruction = "\n\n**Thinking Mode**: You can use <think> tags." if lang_key == 'en' else "\n\n**思考模式**：可以使用<think>标签。"
            system_prompt += thinking_instruction
            
        if context:
            valid_context = [c.strip() for c in context[-3:] if c.strip()]
            if valid_context:
                context_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(valid_context)])
                user_prompt = prompts['user_with_context'].format(context=context_text, text=text)
            else:
                user_prompt = prompts['user_no_context'].format(text=text)
        else:
            user_prompt = prompts['user_no_context'].format(text=text)
            
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_input], return_tensors="pt")
        if self.device != 'cpu':
            model_inputs = model_inputs.to(self.device)
            
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=128,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        refined_text = response.strip()
        # Clean up <think> tags if present
        if '<think>' in refined_text and '</think>' in refined_text:
             refined_text = refined_text.split('</think>')[-1].strip()
        elif '<think>' in refined_text:
             # Assuming incomplete think block or just start
             pass 
             
        # Remove quotes
        if refined_text.startswith('"') and refined_text.endswith('"'): refined_text = refined_text[1:-1]
        
        return refined_text
