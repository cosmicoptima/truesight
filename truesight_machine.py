from dotenv import load_dotenv
load_dotenv()

from abc import ABC, abstractmethod
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from itertools import chain
from math import exp
from numpy import random
import os
import sys
from typing import Generator, List
from yaml import safe_load

from openai import AsyncOpenAI
from rich.console import Console

console = Console(highlight=False)
openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), organization=os.getenv("OPENAI_ORG"))


@dataclass
class Task(ABC):
    @abstractmethod
    def as_text(self, rich=False):
        ...

    @staticmethod
    @abstractmethod
    def from_dict(d):
        ...
    
    def __hash__(self):
        return hash(str(self))

    def __str__(self):
        return self.as_text(rich=False)


@dataclass
class MultipleChoiceTask(Task):
    description_title: str
    description_body: str

    rating_descriptions: OrderedDict[str, str]
    rating_scores: OrderedDict[str, int]

    answers_header: str
    seed_question_file: str

    def as_text(self, rich=False):
        if rich:
            rating_scale = "\n".join([f"[yellow]{rating}:[/yellow] {desc}" for rating, desc in self.rating_descriptions.items()])
            return f"[b]Current task: {self.description_title}.\n\n{self.description_body}[/b]\n\n{rating_scale}\n"
        else:
            rating_scale = "\n".join([f"{rating}: {desc}" for rating, desc in self.rating_descriptions.items()])
            return f"Current task: {self.description_title}.\n\n{self.description_body}\n\n{rating_scale}\n\n{self.answers_header}"

    def ratings(self):
        return list(self.rating_descriptions.keys())
    
    def rating_tokens(self):
        return [f" {token}" for token in self.ratings()]

    @staticmethod
    def from_dict(d):
        return MultipleChoiceTask(
            d["description_title"],
            d["description_body"],
            OrderedDict(d["rating_descriptions"]),
            OrderedDict(d["rating_scores"]),
            d["answers_header"],
            d["seed_question_file"]
        )
    
    __hash__ = Task.__hash__


@dataclass
class FreeResponseTask(Task):
    description_title: str
    description_body: str
    answers_header: str
    seed_question_file: str

    def as_text(self, rich=False):
        if rich:
            return f"[b]Current task: {self.description_title}.\n\n{self.description_body}[/b]\n"
        else:
            return f"Current task: {self.description_title}.\n\n{self.description_body}\n"

    @staticmethod
    def from_dict(d):
        return FreeResponseTask(
            d["description_title"],
            d["description_body"],
            d["answers_header"],
            d["seed_question_file"],
        )
    
    __hash__ = Task.__hash__


def make_task_from_dict(d):
    if d["task_type"] == "multiple_choice":
        return MultipleChoiceTask.from_dict(d)
    elif d["task_type"] == "free_response":
        return FreeResponseTask.from_dict(d)
    else:
        raise ValueError(f"Unknown task type: {d['task_type']}")


with open("assets/tasks.yaml") as f:
    TASKS = {name: make_task_from_dict(d) for name, d in safe_load(f.read()).items()}


class QuestionSequence(ABC):
    @staticmethod
    @abstractmethod
    def from_dict(d):
        ...


@dataclass
class PredefinedQS(QuestionSequence):
    task: Task
    length: int
    initial_pool_size: int
    iterated_pool_size: int
    color: str

    @staticmethod
    def from_dict(d):
        return PredefinedQS(
            TASKS[d["task"]],
            d["length"],
            d["initial_pool_size"],
            d["iterated_pool_size"],
            d["color"]
        )


@dataclass
class WeavedQS(QuestionSequence):
    task: Task
    length: int
    n: int
    color: str

    @staticmethod
    def from_dict(d):
        return WeavedQS(
            TASKS[d["task"]],
            d["length"],
            d["n"],
            d["color"]
        )


def make_question_sequence_from_dict(d):
    if d["type"] == "predefined":
        return PredefinedQS.from_dict(d)
    elif d["type"] == "weaved":
        return WeavedQS.from_dict(d)
    else:
        raise ValueError(f"Unknown question sequence type: {d['qs_type']}")


def make_question_metasequence_from_dict(d):
    sequences = [make_question_sequence_from_dict(qs) for qs in d["sequences"]]

    if d["type"] == "sequence":
        yield from sequences
    elif d["type"] == "cycle":
        while True:
            yield from sequences
    else:
        raise ValueError(f"Unknown question metasequence type: {d['type']}")


def make_question_patasequence_from_dict(d):
    metasequences = [make_question_metasequence_from_dict(metasequence) for metasequence in d]
    for metasequence in metasequences:
        yield from metasequence


with open("question_patasequence.yaml") as f:
    QUESTION_PATASEQUENCE = make_question_patasequence_from_dict(safe_load(f.read()))


@dataclass
class QAPair:
    question: str
    answer: str

    def __str__(self):
        return f"{self.question}: {self.answer}"


class Divination:
    items: List[str]
    seed_questions: List[str]
    question_metasequence: Generator[QuestionSequence, None, None]

    def __init__(self, question_metasequence):
        self.items = []
        self.question_metasequence = question_metasequence

    def indices(self, item):
        task_i = 0
        question_i = 0

        for item_ in self.items[:self.items.index(item) + 1]:
            if isinstance(item_, Task):
                task_i += 1
                question_i = 0
            else:
                question_i += 1
        
        return task_i, question_i

    def latest_task(self):
        return next(item for item in reversed(self.items) if isinstance(item, Task))

    # TODO: it only makes sense to call this method when every multiple-choice task is identical
    def latest_mc_task(self):
        return next(item for item in reversed(self.items) if isinstance(item, MultipleChoiceTask))
    
    def has_been_asked(self, question):
        return question in [item.question for item in self.items if isinstance(item, QAPair)]

    # TODO: `free_response_prompt` and `postfill_prompt` are very similar, refactor

    def prompt(self, question=None):
        prompt = "𓂀 TRUESIGHT MACHINE 𓂀\nMade by Celeste\n\n"
        
        question = f"{question}:" if question is not None else ""
        prompt += "".join([str(item) + "\n" for item in self.items]) + question

        return prompt

    def free_response_prompt(self, question=None):
        prompt = "𓂀 TRUESIGHT MACHINE 𓂀\nMade by Celeste\n\n"

        for item in self.items:
            if isinstance(item, Task):
                prompt += str(item) + "\n"
            elif isinstance(item, QAPair):
                prompt += f"Q: {item.question}\nA: {item.answer}\n\n"
            else:
                raise ValueError(f"Unknown item type: {item}")

        prompt += f"Q:"

        if question is not None:
            prompt += f" {question}\nA:"
        
        return prompt
    
    def postfill_prompt(self, task_i, question_i, question=None):
        prompt = "<|fim_prefix|>𓂀 TRUESIGHT MACHINE 𓂀\nMade by Celeste\n\n"

        for i, item in enumerate(self.items):
            current_task_i, current_question_i = self.indices(item)

            if isinstance(item, Task):
                prompt += str(item) + "\n"
            elif isinstance(item, QAPair):
                prefix = f"(T{current_task_i}Q{current_question_i})"
                prompt += f"{prefix} [Question] {item.question}\n{prefix} [Response] {item.answer}\n"
            else:
                raise ValueError(f"Unknown item type: {item}")
            
            if i < len(self.items) - 1 and not isinstance(self.items[i + 1], Task):
                prompt += "---\n"

        if question is not None:
            current_task_i, current_question_i = self.indices(self.items[-1])
            current_question_i += 1

            prefix = f"(T{current_task_i}Q{current_question_i})"
            prompt += f"{prefix} [Question] {question}\n"
        
        prompt += f"<|fim_suffix|>(T{task_i}Q{question_i}) [Response]"

        return prompt

    def add_task(self, task):
        console.print(task.as_text(rich=True))

        self.items.append(task)
        with open(f"assets/seed_questions/{task.seed_question_file}.txt") as f:
            self.seed_questions = f.read().splitlines()
    
    def add_qa_pair(self, question, answer):
        self.items.append(QAPair(question, answer))
    
    def rank_qa_pairs(self, qa_pairs):
        qa_pairs = random.permutation([qa_pair for qa_pair in qa_pairs if not self.has_been_asked(qa_pair.question)])
        scores = {}

        for qa_pair in qa_pairs:
            rating = qa_pair.answer.split("#")[0].strip()
            if rating in self.latest_mc_task().rating_scores:
                scores[qa_pair.question] = self.latest_mc_task().rating_scores[rating]
            else:
                scores[qa_pair.question] = 0
        
        sorted_qa_pairs = sorted(qa_pairs, key=lambda qa_pair: scores[qa_pair.question], reverse=True)
        return list(set([qa_pair.question for qa_pair in sorted_qa_pairs]))

    async def rank_mc(self, questions):
        async def rank_batch(batch):
            prompts = [self.prompt(question) for question in batch]
            response = await openai.completions.create(model="gpt-4-base", prompt=prompts, max_tokens=1)
            return [QAPair(batch[i], choice.text.strip()) for i, choice in enumerate(response.choices)]
        
        batch_results = await asyncio.gather(*[rank_batch(questions[i:i + 128]) for i in range(0, len(questions), 128)])
        return self.rank_qa_pairs(list(chain(*batch_results)))

    async def rank_fr(self, questions):
        task_i = self.indices(self.items[-1])[0] + 1
        question_i = 1

        prompts = [self.postfill_prompt(task_i, question_i, question) for question in questions]
        response = await openai.completions.create(
            model="gpt-4-base",
            prompt=prompts,
            max_tokens=1,
            logprobs=5,
        )

        scores = {}
        for i, choice in enumerate(response.choices):
            scores[questions[i]] = 0

            for token in choice.logprobs.tokens:
                score = self.latest_mc_task().rating_scores.get(token.strip(), 0)
                p = exp(choice.logprobs.top_logprobs[0].get(token, 0))

                scores[questions[i]] += score * p
        
        return sorted(questions, key=scores.get, reverse=True)
    
    async def rank(self, questions, task):
        if isinstance(task, MultipleChoiceTask):
            return await self.rank_mc(questions)
        else:
            return await self.rank_fr(questions)

    async def weave_mc(self, n):
        response = await openai.completions.create(
            model="gpt-4-base",
            prompt=self.prompt(),
            n=n,
            max_tokens=64,
            top_p=0.97,
            stop=["\n", " #", "#"],
        )

        qa_pairs = []
        for choice in response.choices:
            try:
                question, answer = [text.strip() for text in choice.text.split(":")]
            except ValueError:
                continue

            qa_pairs.append(QAPair(question, answer))

        return self.rank_qa_pairs(qa_pairs)
    
    async def weave_fr(self, n):
        response = await openai.completions.create(
            model="gpt-4-base",
            prompt=self.free_response_prompt(),
            n=n,
            max_tokens=512,
            top_p=0.97,
            stop=["\n"],
        )

        return await self.rank_fr([choice.text.strip() for choice in response.choices])
    
    async def weave(self, n, task):
        if isinstance(task, MultipleChoiceTask):
            return await self.weave_mc(n)
        elif isinstance(task, FreeResponseTask):
            return await self.weave_fr(n)
        else:
            raise ValueError(f"Unknown task type: {type(task)}")
    
    async def ask_question_sequence(self, question_sequence):
        try:
            if hash(question_sequence.task) != hash(self.latest_task()):
                self.add_task(question_sequence.task)
        except StopIteration:
            self.add_task(question_sequence.task)

        match question_sequence:
            case PredefinedQS(task=task, initial_pool_size=initial_pool_size, iterated_pool_size=iterated_pool_size):
                initial_pool = random.choice(self.seed_questions, initial_pool_size, replace=False)
                iterated_pool = (await self.rank(initial_pool, task))[:iterated_pool_size]

        for i in range(question_sequence.length):
            questions = []
            find_new_question = True

            while True:
                if find_new_question:
                    if len(questions) == 0:
                        match question_sequence:
                            case PredefinedQS(task=task):
                                questions = await self.rank(iterated_pool, task)
                            case WeavedQS(task=task, n=n):
                                questions = await self.weave(n, task)
                    
                    question = questions.pop(0)
                    find_new_question = False
                
                color = question_sequence.color
                length = question_sequence.length
                answer = console.input(f"({i + 1}/{length}) [{color}]{question.rstrip('?')}?[/{color}] ")

                if len(answer) > 0:
                    if isinstance(task, MultipleChoiceTask):
                        answer = answer[0].upper() + answer[1:]
                        rating = [text.strip() for text in answer.split("#")][0]

                        if rating in task.ratings():
                            if isinstance(question_sequence, PredefinedQS):
                                if question in self.seed_questions:
                                    self.seed_questions.remove(question)
                                iterated_pool.remove(question)

                            if rating == "Z": # TODO: this may not apply to all tasks
                                find_new_question = True
                            else:
                                break

                    elif isinstance(task, FreeResponseTask):
                        if isinstance(question_sequence, PredefinedQS):
                            if question in self.seed_questions:
                                self.seed_questions.remove(question)
                            iterated_pool.remove(question)
                        
                        if answer.split("#")[0].strip() == "Z": # TODO lol
                            find_new_question = True
                        else:
                            break

                    else:
                        raise ValueError(f"Unknown task type: {type(task)}")

                sys.stdout.write('\033[A\033[K')
            
            self.add_qa_pair(question, answer)
        
        console.print()
    
    async def run(self):
        console.print("\n[red b]𓂀  TRUESIGHT MACHINE 𓂀[/red b]\n")
        for question_sequence in self.question_metasequence:
            await self.ask_question_sequence(question_sequence)


async def main():
    await Divination(QUESTION_PATASEQUENCE).run()


if __name__ == "__main__":
    asyncio.run(main())