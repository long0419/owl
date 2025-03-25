from datasets import Dataset, DatasetDict
from tqdm import tqdm

import json
import random
import argparse
import pandas as pd


def construct_system_message(task_prompt):
    r"""Construct the assistant system message."""
    
    return f"""
===== RULES OF ASSISTANT =====
Never forget you are a assistant and I am a user. Never flip roles! Never instruct me! You have to utilize your available tools to solve the task I assigned.
We share a common interest in collaborating to successfully complete a complex task.
You must help me to complete the task.

Here is our overall task: {task_prompt}. Never forget our task!

I must instruct you based on your expertise and my needs to complete the task. An instruction is typically a sub-task or question.

You must leverage your available tools, try your best to solve the problem, and explain your solutions.
Unless I say the task is completed, you should always start with:
Solution: [YOUR_SOLUTION]
[YOUR_SOLUTION] should be specific, including detailed explanations and provide preferable detailed implementations and examples and lists for task-solving.

Please note that our overall task may be very complicated. Here are some tips that may help you solve the task:
<tips>
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- When trying to solve math problems, you can try to write python code and use sympy library to solve the problem.
- Always verify the accuracy of your final answers! Try cross-checking the answers by other ways. (e.g., screenshots, webpage analysis, etc.).  
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- After writing codes, do not forget to run the code and get the result. If it encounters an error, try to debug it.
- When a tool fails to run, or the code does not run correctly, never assume that it returns the correct result and continue to reason based on the assumption, because the assumed result cannot lead you to the correct answer. The right way is to think about the reason for the error and try again.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- For downloading files, you can either use the web browser simulation toolkit or write codes.
</tips>

    """
    

def process_and_push(input_file, repo_id):
    r"""Process dataset and push to Hugging Face Hub."""
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)

    filtered = [entry for entry in data if entry.get("score", False) is True]
    for entry in tqdm(filtered, desc="Processing"):
        if isinstance(entry.get("history"), list):
            entry["history"] = json.dumps(entry["history"])

    df = pd.DataFrame(filtered)

    def build_messages(history, question):
        r"""Convert history to structured message format."""
        try:
            history = json.loads(history) if isinstance(history, str) else history
        except json.JSONDecodeError:
            return []

        messages = [{"role": "system", 
                     "content": construct_system_message(question)}]
        messages.append({"role": "user", "content": 
                         "We share a common interest in collaborating to "
                         "successfully complete a task."})
        init_prompt = ("Now please give me instructions to solve the overall "
                       "task step by step. If the task requires some specific "
                       "knowledge, please instruct me to use tools to complete "
                       "the task.")
        messages.append({"role": "assistant", "content": init_prompt})

        for entry in history:
            if isinstance(entry, dict):
                user_msg = {"role": "user", "content": entry.get("user", "")}
                assistant_msg = {"role": "assistant", 
                                 "content": entry.get("assistant", "")}
                messages.extend([user_msg, assistant_msg])
        return messages

    df["messages_raw"] = df["history"].apply(
        lambda x: json.dumps(json.loads(x)) if isinstance(x, str) else json.dumps(x)
    )
    df["messages"] = df.apply(
        lambda row: build_messages(row["history"], row["question"]), axis=1)
    df["messages_camel"] = df["messages"]
    
    test_df = df.sample(n=min(5, len(df)), random_state=42)
    train_df = df.drop(test_df.index)
    
    hf_dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df), 
        "test": Dataset.from_pandas(test_df),
    })

    hf_dataset.push_to_hub(repo_id)
    print(f"Dataset pushed to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Process and push dataset.")
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="Path to the input JSON file")
    parser.add_argument("--repo_id", "-r", type=str, required=True,
                        help="Hugging Face Hub repository ID")
    args = parser.parse_args()

    random.seed(42)
    process_and_push(args.input_file, args.repo_id)


if __name__ == "__main__":
    main()
