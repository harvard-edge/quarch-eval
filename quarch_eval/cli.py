import fire
from .run_all_judges import run as judges_main
from .run_all_mcq_evals import run as mcq_main
from .test_llm import run as test_llm_main
from .report import run as report_main
from .list_models import run as list_models_main

def main():
    fire.Fire({
        'judges': judges_main,
        'mcq': mcq_main,
        'test-llm': test_llm_main,
        'report': report_main,
        'list-models': list_models_main,
    })

if __name__ == '__main__':
    main()
