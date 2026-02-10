import argparse
import os
import networkx as nx
import subprocess

from sera.config_schema import GenerateConfig
from sera.datagen.data.generate.classes import LocalRepository, ExistingRepository, RepositoryInstance
from sera.datagen.data.generate.no_bug_dataset import NoBugDataset
from sera.utils import ExperimentFolder

def main(config: GenerateConfig, metadata_dir: str, folder: ExperimentFolder):
    local_repository_list = []
    # Create repositories and containers
    for repo in config.personal_repos:
        local_repository_instance = LocalRepository(org_name=repo.org_name,
                                                    last_name=repo.last_name,
                                                    commits=repo.commits,
                                                    install_cmds=repo.install_cmds,
                                                    test_cmd=repo.test_cmd,
                                                    skip_package_name=repo.skip_package_name,
                                                    python_version=repo.python_version,
                                                    node_version=repo.node_version,
                                                    package_manager=repo.package_manager,
                                                    test_framework=repo.test_framework,
                                                    tsc_verify=repo.tsc_verify,
                                                    top_level_folder=repo.top_level_folder,
                                                    overwrite_cg=repo.overwrite_cg,
                                                    language=repo.language)
        local_repository_instance.setup(repo_parent_dir=config.repo_parent_dir,
                                        n_commits=repo.n_commits,
                                        lookback=repo.lookback,
                                        docker_org=config.docker.docker_org,
                                        gh_mirror_org=config.docker.gh_mirror_org,
                                        metadata_dir=metadata_dir,
                                        max_folder_depth=repo.max_folder_depth)
        local_repository_list.append(local_repository_instance)
    for repo in config.existing_repos:
        existing_repository_instance = ExistingRepository(org_name=repo.org_name,
                                                            last_name=repo.last_name,
                                                            source=repo.source,
                                                            base_commit=repo.base_commit,
                                                            image_name=repo.image_name,
                                                            instance_id=repo.instance_id,
                                                            top_level_folder=repo.top_level_folder,
                                                            overwrite_cg=repo.overwrite_cg,)
        existing_repository_instance.setup(repo_parent_dir=config.repo_parent_dir,
                                            metadata_dir=metadata_dir,
                                            max_folder_depth=repo.max_folder_depth)
        local_repository_list.append(existing_repository_instance)

    dataset = NoBugDataset(config=config, 
                            repositories=local_repository_list,
                            metadata_dir=metadata_dir, 
                            folder=folder,)
    dataset.build_dataset()