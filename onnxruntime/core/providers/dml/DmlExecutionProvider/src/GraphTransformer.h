// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
// Lotus framework headers for onnxruntime::IExecutionProvider (not part of the operator ABI).
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/graph_transformer.h"

namespace Dml
{
    // Applies transforms to a Lotus graph. The graph transformer is responsible for setting the execution provider
    // on the graph nodes which DML supports.
    class GraphTransformer : public onnxruntime::GraphTransformer
    {
    public:
        GraphTransformer(const std::string& name, std::shared_ptr<onnxruntime::KernelRegistry> dmlRegistry);

    private:
        onnxruntime::common::Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level = 0) const final;

    private:
        void PerformOperatorFusion(onnxruntime::Graph* graph, bool* modified) const;
        std::shared_ptr<onnxruntime::KernelRegistry> m_registry;
    };

} // namespace Dml
