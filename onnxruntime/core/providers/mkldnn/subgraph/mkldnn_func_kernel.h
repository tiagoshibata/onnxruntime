// Copyright(C) 2018 Intel Corporation
// Licensed under the MIT License
#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/framework/func_api.h"
#include "mkldnn_kernel.h"

namespace onnxruntime {
namespace mkl_dnn {

namespace {
struct SubgraphParams {
  NodeAttributes attributes;
  MKLDNNExecutionProvider* provider;
  std::shared_ptr<Subgraph> subgraph;
  std::string subgraph_id;
  std::string subgraph_key;

  SubgraphParams() {}
};
}  // namespace

template <typename T>
class MkldnnFuncKernel {
 public:
  explicit MkldnnFuncKernel(const ComputeContext* context,
                            const NodeAttributes& attributes,
                            MKLDNNExecutionProvider* provider) {
    ORT_UNUSED_PARAMETER(context);

    params_.provider = provider;
    params_.attributes = attributes;

    auto sub_it = attributes.find("subgraph_id");
    if (sub_it->second.type() == ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      params_.subgraph_id = sub_it->second.s();
      params_.subgraph = provider->GetMklDnnSubgraph(params_.subgraph_id);

      std::ostringstream key_os;
      key_os << params_.subgraph->graph_name << "_" << params_.subgraph_id << "-";
      key_os << params_.subgraph->mkldnn_nodes.back().ToString() << "-";
      key_os << params_.subgraph->mkldnn_nodes.back().output_name;

      if (params_.subgraph->mkldnn_nodes[0].name == "Conv") {
        std::ostringstream os;
        os << "Conv-" << params_.subgraph->mkldnn_nodes[0].node_index << "-";
        key_os << GetConvAttributeKey(attributes, os.str());
      }

      if (params_.subgraph->mkldnn_nodes[0].name == "LRN") {
        std::ostringstream os;
        os << "LRN-" << params_.subgraph->mkldnn_nodes[0].node_index << "-";
        key_os << GetLrnAttributeKey(attributes, os.str());
      }

      if (params_.subgraph->mkldnn_nodes[0].name.find("Pool") != std::string::npos) {
        std::ostringstream os;
        os << params_.subgraph->mkldnn_nodes[0].name << "-" << params_.subgraph->mkldnn_nodes[0].node_index << "-";
        key_os << GetPoolAttributesKey(attributes, os.str());
      }

      params_.subgraph_key = key_os.str();
    }
  }

  std::string GetPoolAttributesKey(const NodeAttributes& attributes,
                                   const std::string attributes_prefix = "") {
    std::string key;

    auto attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end()) {
      key.append(attr->second.s());
    }

    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "count_include_pad");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(std::to_string(proto.i()));
      key.append(1, '_');
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "ceil_mode");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(std::to_string(proto.i()));
      key.append(1, '_');
      key.append(1, '#');
    }
    return key;
  }

  std::string GetConvAttributeKey(const NodeAttributes& attributes,
                                  const std::string attributes_prefix = "") {
    std::string key;

    auto attr = attributes.find(attributes_prefix + "dilations");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(proto.s());
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      for (int i = 0; i < proto.ints_size(); i++) {
        key.append(std::to_string(proto.ints(i)));
        key.append(1, '_');
      }
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "group");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(std::to_string(proto.i()));
      key.append(1, '#');
    }

    return key;
  }

  std::string GetLrnAttributeKey(const NodeAttributes& attributes,
                                 const std::string attributes_prefix = "") {
    std::string key;

    auto attr = attributes.find(attributes_prefix + "alpha");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(std::to_string(proto.f()));
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "beta");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(std::to_string(proto.f()));
      key.append(1, '#');
    }

    attr = attributes.find(attributes_prefix + "bias");
    if (attr != attributes.end()) {
      key.append(1, '#');
      ONNX_NAMESPACE::AttributeProto proto = attr->second;
      key.append(std::to_string(proto.f()));
      key.append(1, '#');
    }

    return key;
  }

  Status Compute(const OrtCustomOpApi* api, OrtKernelContext* context) const;

 private:
  SubgraphParams params_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime