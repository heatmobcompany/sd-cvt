
from enum import Enum, unique
import time
from helper.logging import Logger
from modules.api import api
logger = Logger("CVT")

@unique
class CvtStatus(Enum):
    Pending = "Pending"
    Active = "Active"
    Completed = "Completed"
    Failed = "Failed"

tasks_info = {}

def add_task(id_job):
    if id_job in tasks_info:
        logger.warning(f"Task {id_job} already exists")
    if len(tasks_info) > 100:
        tasks_info.pop(list(tasks_info.keys())[0])
        
    tasks_info[id_job] = {
        "status": CvtStatus.Pending,
        "progress": 0,
        "result": None,
        "start_time": None,
        "end_time": None
    }

def update_task(task_id, status=None, progress=0, result=None):
    if task_id not in tasks_info:
        logger.error(f"Task {task_id} not found")
        return
    if status:
        tasks_info[task_id]["status"] = status
    if progress:
        tasks_info[task_id]["progress"] = progress
    if result:
        tasks_info[task_id]["result"] = result
    if status == CvtStatus.Active:
        tasks_info[task_id]["start_time"] = time.time()
    elif status == CvtStatus.Completed or status == CvtStatus.Failed:
        tasks_info[task_id]["end_time"] = time.time()

def get_task(task_id):
    if task_id not in tasks_info:
        logger.error(f"Task {task_id} not found")
        return None
    return tasks_info[task_id]