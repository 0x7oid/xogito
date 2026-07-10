'''
Conceptually this is more like a database
it has 4 containers
- Task graph : the tasks with the dependencies (logic to be implmented through the /graph folder and stored here)
- Artifact store : outputs the executors produce , each with evidence attached to it
- Belief table : the claims and their current belief ladder position
- Provenance log : who wrote that and  when and based on what
'''

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from core.kernel import assert_legal_transition


task_kinds = Literal["investigate", "produce", "verify"]
task_statuses = Literal["pending", "in_progress", "completed", "failed" , "rejected"]

@dataclass
class Task:
    id: int
    description: str = ""
    kind: task_kinds
    depends_on : list[int] = field(default_factory=list) # that is a list of ids of the tasks
    status : task_statuses = "pending"
    done_when : str = "" # this is a description of the expected output of the task
    why_now : str = "" # this is a description of the urgency of the task
    rejection_reason : str = "" # this is a description of the reason for rejection of the task


belief_ladder = Literal["unverified",
    "supported",
    "verified",
    "contested"
    ]


@dataclass
class Claim:
    id: int
    statement: str = ""
    belief : belief_ladder = "unverified"
    evidence_ids : list[int] = field(default_factory=list) # that is a list of ids of the artifacts

@dataclass
class Artifact:
    id: int
    task_id: int
    content: str = ""
    summary : str = "" # to be used i n the planner later to  reduce context costs
    claim_ids : list[int] = field(default_factory=list) # that is a list of ids of the claims
    evidence_ids : list[int] = field(default_factory=list) # that is a list of ids of the artifacts

actor_kinds = Literal[
    "user",
    "planner",
    "scheduler",
    "executor",
    "evaluator",
    "adjudicator",
    "contested",
    "checkpoint",
    "compressor",
]

target_types = Literal[
    "task",
    "claim",
    "artifact",
]

@dataclass
class ProvenanceEntry:
    timestamp: datetime
    actor : actor_kinds # simply this is the agent who initiated the action
    action : str
    target_id : int # this can be for example a task_1 , claim_2, artifact_3 ...
    target_type : target_types


# there is a many to many relationship between claims and artifacts


@dataclass
class Workspace:
    spec : dict
    _tasks: dict[int, Task] = field(default_factory=dict)
    _claims: dict[int, Claim] = field(default_factory=dict)
    _artifacts: dict[int, Artifact] = field(default_factory=dict)
    _provenance_log: list[ProvenanceEntry] = field(default_factory=list)

    next_task_id: int = 0
    next_claim_id: int = 0
    next_artifact_id: int = 0
# LOGGER FUNCTION ########################################################################################################################
    def _log_provenance(self, actor: actor_kinds, action: str, target_id: int, target_type: target_types):
        entry = ProvenanceEntry(
            timestamp=datetime.now(),
            actor=actor,
            action=action,
            target_id=target_id,
            target_type=target_type
        )
        self._provenance_log.append(entry)

# SETTERS AND VALIDATORS ########################################################################################################################
    def validate_task_dependencies(self, task: Task):
        for dependency_id in task.depends_on:
            if dependency_id not in self._tasks:
                raise ValueError(f"Dependency task with id {dependency_id} does not exist.")
            # todo : call for cyclic dependency check here

    def add_task(self, task: Task):
        self.validate_task_dependencies(task)
        task.id = self.next_task_id
        self._tasks[task.id] = task
        self.next_task_id += 1
        self._log_provenance(actor="planner", action="add_task", target_id=task.id, target_type="task")

    def validate_claim_evidence(self, claim: Claim):
        for evidence_id in claim.evidence_ids:
            if evidence_id not in self._artifacts:
                raise ValueError(f"Evidence artifact with id {evidence_id} does not exist.")
            
    def add_claim(self, claim: Claim):
        self.validate_claim_evidence(claim)
        claim.id = self.next_claim_id
        self._claims[claim.id] = claim
        self.next_claim_id += 1
        self._log_provenance(actor="planner", action="add_claim", target_id=claim.id, target_type="claim")

    def validate_artifact_claims(self, artifact: Artifact):
        for claim_id in artifact.claim_ids:
            if claim_id not in self._claims:
                raise ValueError(f"Claim with id {claim_id} does not exist.")
        for evidence_id in artifact.evidence_ids:
            if evidence_id not in self._artifacts:
                raise ValueError(f"Evidence artifact with id {evidence_id} does not exist.")
        for task_id in [artifact.task_id]:
            if task_id not in self._tasks:
                raise ValueError(f"Task with id {task_id} does not exist.")
            

            
    def add_artifact(self, artifact: Artifact):
        self.validate_artifact_claims(artifact)
        artifact.id = self.next_artifact_id
        self._artifacts[artifact.id] = artifact
        self.next_artifact_id += 1
        self._log_provenance(actor="executor", action="add_artifact", target_id=artifact.id, target_type="artifact")

    def validate_provenance_entry(self, entry: ProvenanceEntry):
        if entry.target_type == "task" and entry.target_id not in self._tasks:
            raise ValueError(f"Task with id {entry.target_id} does not exist.")
        elif entry.target_type == "claim" and entry.target_id not in self._claims:
            raise ValueError(f"Claim with id {entry.target_id} does not exist.")
        elif entry.target_type == "artifact" and entry.target_id not in self._artifacts:
            raise ValueError(f"Artifact with id {entry.target_id} does not exist.")
        
    def add_provenance_entry(self, entry: ProvenanceEntry):
        self.validate_provenance_entry(entry)
        self._provenance_log.append(entry)

    def link_evidence(self , claim_id:int , artifact_id:int):
        # a bidirectional link between a claim and an artifact
        # without this function we would have a referential integrity violation between a claim and the artifact
        if claim_id not in self._claims:
            raise ValueError(f"Claim with id {claim_id} does not exist.")
        if artifact_id not in self._artifacts:
            raise ValueError(f"Artifact with id {artifact_id} does not exist.")
        
        self._claims[claim_id].evidence_ids.append(artifact_id)
        self._artifacts[artifact_id].claim_ids.append(claim_id)
    def unlink_evidence(self , claim_id:int , artifact_id:int):
        # a bidirectional unlink between a claim and an artifact
        # without this function we would have a referential integrity violation between a claim and the artifact
        if claim_id not in self._claims:
            raise ValueError(f"Claim with id {claim_id} does not exist.")
        if artifact_id not in self._artifacts:
            raise ValueError(f"Artifact with id {artifact_id} does not exist.")
        
        self._claims[claim_id].evidence_ids.remove(artifact_id)
        self._artifacts[artifact_id].claim_ids.remove(claim_id)

    def update_belief_of_claim(self, claim_id: int, new_belief: belief_ladder , veridict : dict , actor: actor_kinds):
        # in order to update the belief we need a veridct confirmation from the kernel
        if claim_id not in self._claims:
            raise ValueError(f"Claim with id {claim_id} does not exist.")
        
        old_belief = self._claims[claim_id].belief

        for evidence_id in veridict.get("evidence_ids", []):
            if evidence_id not in self._artifacts:
                raise ValueError(f"Evidence artifact with id {evidence_id} does not exist.")
    
        assert_legal_transition(old_belief, new_belief , veridict)


        self._claims[claim_id].belief = new_belief
        self._log_provenance(actor="evaluator", action=f"update_belief_to_{new_belief}", target_id=claim_id, target_type="claim")


    def update_task_status(self, task_id: int, new_status: task_statuses):
        if task_id not in self._tasks:
            raise ValueError(f"Task with id {task_id} does not exist.")
        self._tasks[task_id].status = new_status
        self._log_provenance(actor="executor", action=f"update_task_status_to_{new_status}", target_id=task_id, target_type="task")

# GETTERS ########################################################################################################################
    def get_task(self, task_id: int) -> Task:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise ValueError(f"Task with id {task_id} does not exist.")
        

    def get_claim(self, claim_id: int) -> Claim:
        try:
            return deepcopy(self._claims[claim_id]) # protection mechanism to avoid external mutation of the internal state
        except KeyError:
            raise ValueError(f"Claim with id {claim_id} does not exist.")
        
    def get_artifact(self, artifact_id: int) -> Artifact:
        try:
            return self._artifacts[artifact_id]
        except KeyError:
            raise ValueError(f"Artifact with id {artifact_id} does not exist."
                             )
    def get_provenance(self , target_type: target_types, target_id: int) -> list[ProvenanceEntry]:
        return [entry for entry in self._provenance_log if entry.target_type == target_type and entry.target_id == target_id]
    
    def evidence_for_claim(self, claim_id:int):
        if claim_id not in self._claims:
            raise ValueError(f"Claim with id {claim_id} does not exist.")
        evidence_ids = self._claims[claim_id].evidence_ids
        return deepcopy([self.get_artifact(evidence_id) for evidence_id in evidence_ids])
    
    def snapshot(self):
        # this function will be used by the executor
        return {
            "tasks": deepcopy(list(self._tasks.values())),
            "claims": deepcopy(list(self._claims.values())),
            "artifacts": deepcopy(list(self._artifacts.values())),
            "provenance_log": deepcopy(self._provenance_log),
        }
    


